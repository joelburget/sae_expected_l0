## Running

```
> wandb sweep --project sae-expected-l0-sweep-norm gpt2-config.yaml
> wandb agent --count 10 <sweep id printed by previous command>
```

If you have more GPU memory:

```
> wandb sweep --project sae-expected-l0-sweep-norm gemma-config.yaml
```
