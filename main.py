import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from transformer_lens import HookedTransformer
from datasets import load_dataset
import wandb

# TODO: Gemma-2B
model_name = "gpt2"
ds_name = "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"
expansion_factor = 8
training_tokens = 8_000_000
hook_point = "blocks.6.hook_resid_post"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained(model_name, device=device)
ds = load_dataset(ds_name)
input_dim = model.cfg.d_model
normal = Normal(0, 1)


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


# From [Tutorial: Implementing and Training JumpReLU SAEs.ipynb](https://colab.research.google.com/drive/1PlFzI_PWGTN9yCQLuBcSuPJUjgHL7GiD?usp=sharing#scrollTo=Zz0TwpNCuDOP)
def remove_parallel_component(x, v):
    """Returns x with component parallel to v projected away."""
    v_normalised = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)
    parallel_component = torch.einsum("...d,...d->...", x, v_normalised)
    return x - parallel_component[..., None] * v_normalised

def train(config=None):
    with wandb.init(config=config):

        activation = None
        def save_activation_hook(module, input, output):
            nonlocal activation
            activation = output
        model.get_submodule(hook_point).register_forward_hook(save_activation_hook)

        config = wandb.config
        learning_rate = config.learning_rate
        reconstruction_coefficient = config.reconstruction_coefficient
        stddev_prior = config.stddev_prior
        hidden_dim = input_dim * expansion_factor

        sae = SparseAutoencoder(input_dim, hidden_dim, stddev_prior)
        sae.to(device)
        optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

        i, total_tokens = 0, 0
        for input in ds["train"]:
            try:
                tokens = input["input_ids"]
                total_tokens += len(tokens)
                model(torch.tensor(tokens, device=device))

                x_hat, pre_activation = sae(activation)

                per_item_mse_loss = F.mse_loss(activation, x_hat, reduction="none")
                reconstruction_loss = per_item_mse_loss.sum(dim=-1).mean()
                l0_loss = sae.expected_l0_loss(pre_activation)
                loss = (reconstruction_coefficient * reconstruction_loss + l0_loss) / (reconstruction_coefficient + 1)

                optimizer.zero_grad()
                loss.backward()

                sae.decoder.weight.grad = remove_parallel_component(
                    sae.decoder.weight.grad, sae.decoder.weight
                )
                optimizer.step()
                sae.decoder.weight.data = sae.decoder.weight / sae.decoder.weight.norm(dim=-1, keepdim=True)

                log_info = {
                    "loss": loss.item(),
                    "reconstruction_loss": reconstruction_loss.item(),
                    "l0_loss": l0_loss.item(),
                    "total_tokens": total_tokens,
                }
                if total_tokens > training_tokens:
                    wandb.log(log_info)
                    break
                if i % 10 == 0:
                    wandb.log(log_info)
                i += 1
            except BaseException as e:
                print(e)
                pass

        sae_save_path = "sae.pth"
        torch.save(sae.state_dict(), sae_save_path)
        wandb.save(sae_save_path)
        artifact = wandb.Artifact(
            f"sae-{model_name}-{stddev_prior}-{reconstruction_coefficient}-{hidden_dim}-{learning_rate}",
            type="model",
        )
        artifact.add_file(sae_save_path)


if __name__ == "__main__":
    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        train(config)
