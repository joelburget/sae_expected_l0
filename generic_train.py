from typing import Any, Tuple, Generator
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from transformer_lens import HookedTransformer
from datasets import load_dataset
import wandb
from dataclasses import dataclass, fields
from sae_lens.evals import get_eval_everything_config, run_evals
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.sae import SAE as SaeLensSAE, SAEConfig
from sae_lens.config import LanguageModelSAERunnerConfig
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
normal = Normal(0, 1)
batch_size = 4096  # copied from sweep-gpt2.py

eval_config = get_eval_everything_config()


@dataclass
class SweepConfig:
    model_name: str
    dataset_name: str
    dataset_is_tokenized: bool
    context_size: int
    hook_point: str
    hook_layer: int
    config_path: str
    expansion_factor: int
    training_tokens: int


@dataclass
class TrainConfig(SweepConfig):
    # We sweep these three params
    stddev_prior: float
    learning_rate: float
    reconstruction_coefficient: float

    @classmethod
    def from_sweep_config(
        cls,
        sweep_config: SweepConfig,
        stddev_prior: float,
        learning_rate: float,
        reconstruction_coefficient: float,
        **kwargs,
    ):
        sweep_dict = {
            field.name: getattr(sweep_config, field.name)
            for field in fields(SweepConfig)
        }
        sweep_dict.update(
            dict(
                stddev_prior=stddev_prior,
                learning_rate=learning_rate,
                reconstruction_coefficient=reconstruction_coefficient,
                **kwargs,
            )
        )
        return cls(**sweep_dict)


def eval_sae(sae, model) -> dict[str, Any]:
    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        streaming=True,
        store_batch_size_prompts=8,
        n_batches_in_buffer=8,
        device=device,
    )
    return run_evals(sae, activation_store, model, eval_config)


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, stddev_prior):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.stddev_prior = stddev_prior

    def forward(self, x):
        pre_activation = self.encoder(x)
        a = F.relu(pre_activation)
        x_hat = self.decoder(a)
        return x_hat, pre_activation

    def expected_l0_loss(self, pre_activation):
        stddevs = self.stddev_prior * torch.sqrt((self.encoder.weight**2).sum(dim=1))
        prob_non_zero = 1 - normal.cdf(-pre_activation / stddevs)
        return prob_non_zero.sum()

    def to_sae_lens(self, sweep_config: SweepConfig):
        d_hidden, d_in = self.encoder.weight.shape
        conf = SAEConfig(
            architecture="standard",
            d_in=d_in,
            d_sae=d_hidden,
            activation_fn_str="relu",
            apply_b_dec_to_input=False,
            finetuning_scaling_factor=False,
            context_size=sweep_config.context_size,  # TODO: what is this? does it matter?
            model_name=sweep_config.model_name,
            hook_name=sweep_config.hook_point,
            hook_layer=sweep_config.hook_layer,
            hook_head_index=None,
            prepend_bos=False,
            dataset_path=sweep_config.dataset_name,
            dataset_trust_remote_code=False,
            normalize_activations=False,
            dtype="bfloat16",
            device=device,
            sae_lens_training_version=None,
        )
        result = SaeLensSAE(conf)
        result.W_enc.data = self.encoder.weight.T
        result.b_enc = self.encoder.bias
        result.W_dec.data = self.decoder.weight.T
        result.b_dec = self.decoder.bias
        return result


# From [Tutorial: Implementing and Training JumpReLU SAEs.ipynb](https://colab.research.google.com/drive/1PlFzI_PWGTN9yCQLuBcSuPJUjgHL7GiD?usp=sharing#scrollTo=Zz0TwpNCuDOP)
def remove_parallel_component(x, v):
    """Returns x with component parallel to v projected away."""
    v_normalised = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)
    parallel_component = torch.einsum("...d,...d->...", x, v_normalised)
    return x - parallel_component[..., None] * v_normalised


def enumerate_tokens(config: SweepConfig) -> Generator[torch.Tensor, None, None]:
    ds = load_dataset(config.dataset_name)

    if config.dataset_is_tokenized:
        for input in ds["train"]:
            yield torch.tensor(input["input_ids"], device=device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        for input in ds["train"]:
            tokens = tokenizer(input["text"], return_tensors="pt")["input_ids"]
            yield tokens.to(device)


def train(config: TrainConfig) -> Tuple[HookedTransformer, SparseAutoencoder]:
    reconstruction_coefficient = config.reconstruction_coefficient
    model = HookedTransformer.from_pretrained(
        config.model_name, device=device, first_n_layers=config.hook_layer + 1
    )
    input_dim = model.cfg.d_model

    hidden_dim = input_dim * config.expansion_factor

    sae = SparseAutoencoder(input_dim, hidden_dim, config.stddev_prior)
    sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)

    activation_store = ActivationsStore.from_config(
        model,
        cfg=LanguageModelSAERunnerConfig(
            model_name=config.model_name,
            hook_name=config.hook_point,
            hook_layer=config.hook_layer,
            dataset_path=config.dataset_name,
            context_size=config.context_size,
            is_dataset_tokenized=config.dataset_is_tokenized,
            prepend_bos=True,
            expansion_factor=config.expansion_factor,
            use_cached_activations=False,
            d_in=input_dim,
            training_tokens=config.training_tokens,
            train_batch_size_tokens=batch_size,
        ),
    )

    i, total_tokens = 0, 0
    while True:
        try:
            activation = activation_store.next_batch()[:, 0, :].to(device)
            total_tokens += batch_size
            x_hat, pre_activation = sae(activation)

            per_item_mse_loss = F.mse_loss(activation, x_hat, reduction="none")
            reconstruction_loss = per_item_mse_loss.sum(dim=-1).mean()
            l0_loss = sae.expected_l0_loss(pre_activation)
            loss = (reconstruction_coefficient * reconstruction_loss + l0_loss) / (
                reconstruction_coefficient + 1
            )

            optimizer.zero_grad()
            loss.backward()

            sae.decoder.weight.grad = remove_parallel_component(
                sae.decoder.weight.grad, sae.decoder.weight
            )
            optimizer.step()
            sae.decoder.weight.data = sae.decoder.weight / sae.decoder.weight.norm(
                dim=-1, keepdim=True
            )

            log_info = dict(
                loss=loss.item(),
                reconstruction_loss=reconstruction_loss.item(),
                l0_loss=l0_loss.item(),
                total_tokens=total_tokens,
            )
            if total_tokens >= config.training_tokens:
                wandb.log(log_info)
                break
            if i % 10 == 0:
                wandb.log(log_info)
            i += 1
        except BaseException as e:
            print(e)
            pass

    return model, sae


def sweep(config: SweepConfig):
    with open(config.config_path) as file:
        wandb_sweep_config = yaml.load(file, Loader=yaml.FullLoader)
        with wandb.init(config=wandb_sweep_config):
            learning_rate = wandb.config.learning_rate
            reconstruction_coefficient = wandb.config.reconstruction_coefficient
            stddev_prior = wandb.config.stddev_prior

            model, sae = train(
                TrainConfig.from_sweep_config(
                    config,
                    stddev_prior=stddev_prior,
                    learning_rate=learning_rate,
                    reconstruction_coefficient=reconstruction_coefficient,
                )
            )

            wandb.log(eval_sae(sae.to_sae_lens(config), model))
            sae_save_path = "sae.pth"
            torch.save(sae.state_dict(), sae_save_path)
            wandb.save(sae_save_path)
            artifact = wandb.Artifact(
                f"sae-{config.model_name}-sd{stddev_prior}-rc{reconstruction_coefficient}-{config.expansion_factor}x-lr{learning_rate}",
                type="model",
            )
            artifact.add_file(sae_save_path)
            artifact.save()
            artifact.wait()
