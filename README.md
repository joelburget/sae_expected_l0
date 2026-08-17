## Idea

$L_0$ loss is not differentiable, so [Rajamanoharan et al.](https://arxiv.org/abs/2407.14435) propose to instead minimize expected $L_0$ loss, given some kernel density estimate of the probability density of neural net activations. This idea does not depend upon jump ReLUs, and in fact, I would argue that there is a very obvious and natural way to apply this regularizing loss to a Sparse Autoencoder (SAE) with normal ReLUs. In this document, I'll try to describe my vision for this.

Say we have a very simple autoencoder: the encoder and decoder are just fully connected layers, 1 layer deep. There's a ReLU activation in the middle.

```math
h = \mathrm{ReLU}(W_1 x + b_1), \qquad \hat{x} = W_2 h + b_2
```

Where $W_1 \in \mathbb{R}^{m \times n}$, $W_2 \in \mathbb{R}^{n \times m}$, $b_1 \in \mathbb{R}^m$, and $b_2 \in \mathbb{R}^n$ are the weight matrices and bias vectors for the encoder and decoder, respectively.

During training, imagine replacing each activation vector seen during training, $x$, with a probability distribution over possible activation vectors.

```math
x \sim \mathcal{N}(x, \sigma^2 I)
```

Let's say that this is an isotropic Gaussian distribution centered at $x$ with standard deviation $\sigma$. $\sigma$ is a fixed hyperparameter of the regularization function.

Since the hidden layer's pre-activations (the activations of the hidden layer before the ReLU) are just a linear combination of the input activations $x$, these pre-activations will also be Gaussian-distributed. It's fast and easy to calculate the distribution these come from:

```math
z_i \sim \mathcal{N}(\mu_i, \sigma_i^2), \qquad \mu_i = (W_1 x + b_1)_i, \qquad \sigma_i = \sigma \left\lVert (W_1)_{i} \right\rVert_2
```

Where $\mu_i$ and $\sigma_i$ are the mean and standard deviation of the $i$-th hidden neuron's pre-activation, respectively.

The probability that the activation will be nonzero is the same as the probability that the pre-activation will be greater than zero.

```math
P(h_i \neq 0) = P(z_i > 0) = \Phi\left(\frac{\mu_i}{\sigma_i}\right)
```

Where $\Phi(z)$ is the cumulative distribution function for a standard normal distribution.

Cumulatively, the expected $L_0$ loss is just the sum of these probabilities. I.e.,

```math
\mathbb{E}\left[\left\lVert h \right\rVert_0\right] = \sum_{i=1}^{m} \Phi\left(\frac{\mu_i}{\sigma_i}\right)
```

This loss function should minimize the expected $L_0$ norm of the hidden activations, given that our kernel density assumption is reasonable.

## Implementation

The idea is straightforward to translate to PyTorch. Search for `l0_loss` and `expected_l0_loss` in generic_train.py.

## Running

```
> wandb sweep --project sae-expected-l0-sweep-norm gpt2-config.yaml
> wandb agent --count 10 <sweep id printed by previous command>
```

If you have more GPU memory:

```
> wandb sweep --project sae-expected-l0-sweep-norm gemma-config.yaml
```
