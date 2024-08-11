import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from datasets import load_dataset

model_name = "roneneldan/TinyStories-1M"
ds_name = "roneneldan/TinyStories"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = HookedTransformer.from_pretrained(model_name, device="cpu")
ds = load_dataset(ds_name)

input_dim = model.cfg.d_model
expansion_factor = 8
hidden_dim = input_dim * expansion_factor
sigma = 1.0
learning_rate = 1e-3
beta = 1e-3
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


sae = SparseAutoencoder(input_dim, hidden_dim, sigma)


def train(model, sae, ds, learning_rate=learning_rate, beta=beta):
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    i = 0
    for input in ds["train"]:
        input = input["text"]
        tokens = tokenizer(input)["input_ids"]
        _, cache = model.run_with_cache(torch.tensor(tokens), remove_batch_dim=True)
        x = cache[hook_point]

        x_hat, h = sae(x)

        reconstruction_loss = criterion(x_hat, x)
        l0_loss = sae.expected_l0_loss(h)
        loss = reconstruction_loss + beta * l0_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        i += 1
        if i % 100 == 0:
            print(
                f"Step [{i}/10000], Loss: {loss.item():.4f}, Reconstruction Loss: {reconstruction_loss.item():.4f}, L0 Loss: {l0_loss.item():.6f}"
            )
            if i % 1_000 == 0:
                break


if __name__ == "__main__":
    train(model, sae, ds)
