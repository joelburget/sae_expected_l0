import yaml
import wandb
import torch
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae_training_runner import SAETrainingRunner

from sweep_gpt2 import config as gpt2_config
from generic_train import batch_size

if __name__ == "__main__":
    with open("gpt2-topk-config.yaml") as config:
        wandb_sweep_config = yaml.load(config, Loader=yaml.FullLoader)
        with wandb.init(config=wandb_sweep_config):
            k = wandb.config.k
            learning_rate = wandb.config.learning_rate

            cfg = LanguageModelSAERunnerConfig(
                architecture="topk",
                activation_fn_kwargs={"k": k},
                model_name=gpt2_config.model_name,
                hook_name=gpt2_config.hook_point,
                hook_layer=gpt2_config.hook_layer,
                d_in=768,
                dataset_path=gpt2_config.dataset_name,
                is_dataset_tokenized=True,
                prepend_bos=True,
                expansion_factor=gpt2_config.expansion_factor,
                training_tokens=gpt2_config.training_tokens,
                train_batch_size_tokens=batch_size,
                lr=learning_rate,
                log_to_wandb=True,
                wandb_project="sae_expected_l0",
                context_size=gpt2_config.context_size,
            )
            sae = SAETrainingRunner(cfg).run()

            sae_save_path = "sae.pth"
            torch.save(sae.state_dict(), sae_save_path)
            wandb.save(sae_save_path)
            artifact = wandb.Artifact(
                f"sae-{config.model_name}-topk-{config.expansion_factor}x-lr{learning_rate}",
                type="model",
            )
            artifact.add_file(sae_save_path)
            artifact.save()
            artifact.wait()
