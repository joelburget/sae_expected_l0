import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tqdm import tqdm
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from datasets import load_dataset
import wandb

model_name = "roneneldan/TinyStories-1M"
ds_name = "roneneldan/TinyStories"

sigmas = [0.01, 0.1, 1.0, 10.0, 100.0]
l0_coefficients = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
expansion_factors = [8]
learning_rates = [5e-6, 1e-5, 1e-4]
training_tokens = 20_000_000

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = HookedTransformer.from_pretrained(model_name, device="cpu")
ds = load_dataset(ds_name)

input_dim = model.cfg.d_model
hook_point = "blocks.4.hook_resid_post"

normal = Normal(0, 1)


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sigma):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sigma = sigma

    def forward(self, x):
        h = self.encoder(x)
        a = F.relu(h)
        x_hat = self.decoder(a)
        return x_hat, h

    def expected_l0_loss(self, h):
        W1 = self.encoder.weight
        mu = h  # (mu is just the pre-activations)
        sigma = self.sigma * torch.sqrt((W1**2).sum(dim=1))
        prob_non_zero = 1 - normal.cdf(-mu / sigma)
        return prob_non_zero.sum()


def train(model, sae, ds, learning_rate, l0_coefficient):
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    i = 0
    total_tokens = 0
    for input in ds["train"]:
        input = input["text"]
        tokens = tokenizer(input)["input_ids"]
        total_tokens += len(tokens)
        _, cache = model.run_with_cache(torch.tensor(tokens), remove_batch_dim=True)
        x = cache[hook_point]

        x_hat, h = sae(x)

        reconstruction_loss = criterion(x_hat, x)
        l0_loss = sae.expected_l0_loss(h)
        loss = reconstruction_loss + l0_coefficient * l0_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            wandb.log(
                {
                    "loss": loss.item(),
                    "reconstruction_loss": reconstruction_loss.item(),
                    "l0_loss": l0_loss.item(),
                    "total_tokens": total_tokens,
                }
            )
        i += 1
        if total_tokens > training_tokens:
            break


if __name__ == "__main__":
    for sigma in tqdm(sigmas, desc="sigma", position=0):
        for l0_coefficient in tqdm(
            l0_coefficients, desc="l0_coefficient", position=1, leave=False
        ):
            for expansion_factor in tqdm(
                expansion_factors, desc="expansion_factor", position=2, leave=False
            ):
                for learning_rate in tqdm(
                    learning_rates, desc="learning_rate", position=3, leave=False
                ):
                    wandb.init(
                        project="sae_expected_l0",
                        # track hyperparameters and run metadata
                        config={
                            "model_name": model_name,
                            "dataset": ds_name,
                            "sigma": sigma,
                            "l0_coefficient": l0_coefficient,
                            "expansion_factor": expansion_factor,
                            "learning_rate": learning_rate,
                        },
                    )
                    hidden_dim = input_dim * expansion_factor
                    sae = SparseAutoencoder(input_dim, hidden_dim, sigma)
                    train(model, sae, ds, learning_rate, l0_coefficient)

                    sae_save_path = "sae.pth"
                    torch.save(sae.state_dict(), sae_save_path)
                    wandb.save(sae_save_path)
                    artifact = wandb.Artifact(
                        f"sae-{sigma}-{l0_coefficient}-{expansion_factor}-{learning_rate}",
                        type="model",
                    )
                    artifact.add_file(sae_save_path)
                    wandb.finish()
