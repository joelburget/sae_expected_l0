import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from datasets import load_dataset
import wandb

model_name = "roneneldan/TinyStories-1M"
ds_name = "roneneldan/TinyStories"

expansion_factor = 8
training_tokens = 8_000_000

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = HookedTransformer.from_pretrained(model_name, device=device)
ds = load_dataset(ds_name)

input_dim = model.cfg.d_model
hook_point = "blocks.4.hook_resid_post"

normal = Normal(0, 1)

sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "l0_coefficient": {"min": 8e-6, "max": 1e-4},  # e.g. 0.0412 / 2000 = 0.00002
        "stddev_prior": {"min": 0.005, "max": 0.04},
        "learning_rate": {"min": 1e-5, "max": 5e-3},
    },
}


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


# From https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-optimization:
# "We find that instead removing any gradient information parallel to our dictionary vectors before applying the gradient step results in a small but real reduction in total loss."
def remove_parallel_gradient(weights):
    parallel_component = (weights.grad * weights).sum(dim=1, keepdim=True) * weights
    weights.grad -= parallel_component


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        learning_rate = config.learning_rate
        l0_coefficient = config.l0_coefficient
        stddev_prior = config.stddev_prior
        hidden_dim = input_dim * expansion_factor

        sae = SparseAutoencoder(input_dim, hidden_dim, stddev_prior)
        sae.to(device)
        optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

        i, total_tokens = 0, 0
        for input in ds["train"]:
            try:
                input = input["text"]
                tokens = tokenizer(input)["input_ids"]
                total_tokens += len(tokens)
                _, cache = model.run_with_cache(
                    torch.tensor(tokens, device=device), remove_batch_dim=True
                )
                x = cache[hook_point]

                x_hat, pre_activation = sae(x)

                per_item_mse_loss = F.mse_loss(x, x_hat, reduction="none")
                reconstruction_loss = per_item_mse_loss.sum(dim=-1).mean()
                l0_loss = sae.expected_l0_loss(pre_activation)
                loss = reconstruction_loss + l0_coefficient * l0_loss

                optimizer.zero_grad()
                loss.backward()

                remove_parallel_gradient(sae.encoder.weight)
                # normalize the *columns* of the encoder weight matrix
                sae.encoder.weight = nn.Parameter(
                    sae.encoder.weight / sae.encoder.weight.norm(dim=0, keepdim=True)
                )

                optimizer.step()

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
            f"sae-{stddev_prior}-{l0_coefficient}-{expansion_factor}-{learning_rate}",
            type="model",
        )
        artifact.add_file(sae_save_path)


if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="sae-expected-l0-sweep-norm"
    )
    wandb.agent(sweep_id, train, count=10)
