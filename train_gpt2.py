import torch
import tqdm
import wandb

from generic_train import eval_sae, train, TrainConfig

if __name__ == "__main__":
    with wandb.init():
        for layer in tqdm.tqdm(range(12)):
            for hook_point in ["hook_resid_pre", "hook_resid_post"]:
                config = TrainConfig(
                    model_name="gpt2",
                    dataset_name="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
                    dataset_is_tokenized=True,
                    hook_point=f"blocks.{layer}.{hook_point}",
                    hook_layer=layer,
                    config_path="gpt2-config.yaml",
                    expansion_factor=8,
                    training_tokens=100_000_000,
                    stddev_prior=0.015,
                    learning_rate=0.001,
                    reconstruction_coefficient=200000000.0,
                )
                model, sae = train(config)

                wandb.log(eval_sae(sae.to_sae_lens(config), model))
                sae_save_path = "sae.pth"
                torch.save(sae.state_dict(), sae_save_path)
                wandb.save(sae_save_path)
                artifact = wandb.Artifact(
                    f"sae-{config.model_name}-{config.expansion_factor}x",
                    type="model",
                )
                artifact.add_file(sae_save_path)
                artifact.save()
                artifact.wait()
