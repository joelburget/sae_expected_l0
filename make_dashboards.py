import os
from pathlib import Path
import wandb
import torch

from generic_train import SparseAutoencoder
from sae_dashboard import sae_vis_runner
from sae_dashboard.data_writing_fns import save_feature_centric_vis
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from sae_lens import ActivationsStore, SAE, SAEConfig, run_evals
from sae_lens.evals import EvalConfig
from sweep_gpt2 import config
from transformer_lens import HookedTransformer

wandb_sweep_id = "cn2k18xr"
wandb_project_name = "sae-expected-l0-sweep-norm"
wandb_entity = "PEAR-ML"
hook_point = config.hook_point
dataset_path = config.dataset_name
device = (
    "mps"
    if torch.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
dataset_size = 15_000


def SAEofSparseAutoencoder(sae: SparseAutoencoder) -> SAE:
    d_hidden, d_in = sae.encoder.weight.shape
    conf = SAEConfig(
        architecture="standard",
        d_in=d_in,
        d_sae=d_hidden,
        activation_fn_str="relu",
        apply_b_dec_to_input=False,
        finetuning_scaling_factor=False,
        context_size=1024,  # TODO: what is this? does it matter?
        model_name="gpt2",
        hook_name=hook_point,
        hook_layer=6,
        hook_head_index=None,
        prepend_bos=False,
        dataset_path=dataset_path,
        dataset_trust_remote_code=False,
        normalize_activations=False,
        dtype="bfloat16",
        device=device,
        sae_lens_training_version=None,
    )
    result = SAE(conf)
    result.W_enc.data = sae.encoder.weight.T
    result.b_enc = sae.encoder.bias
    result.W_dec.data = sae.decoder.weight.T
    result.b_dec = sae.decoder.bias
    return result


if __name__ == "__main__":
    data = load_dataset(dataset_path, split=f"train[:{dataset_size}]")
    assert isinstance(data, Dataset)

    all_tokens = torch.tensor(data["input_ids"])
    assert isinstance(all_tokens, torch.Tensor)

    api = wandb.Api()
    sweep = api.sweep(f"{wandb_entity}/{wandb_project_name}/sweeps/{wandb_sweep_id}")
    save_dir = f"{wandb_sweep_id}-files"
    os.makedirs(save_dir, exist_ok=True)

    model = HookedTransformer.from_pretrained(
        config.model_name, device=device, first_n_layers=config.hook_layer + 1
    )
    input_dim = model.cfg.d_model
    hidden_dim = input_dim * config.expansion_factor

    for run in sweep.runs:
        file_path = os.path.join(save_dir, f"{run.name}_sae.pth")

        try:
            file = run.file("sae.pth")
            file.download(root=save_dir, replace=True)
            downloaded_file_path = os.path.join(save_dir, "sae.pth")
            os.rename(downloaded_file_path, file_path)
            print(f"Downloaded {file_path} from run {run.name}")
        except Exception as e:
            print(f"Failed to download sae.pth from run {run.name}: {str(e)}")

    for run in sweep.runs:
        try:
            file_path = os.path.join(save_dir, f"{run.name}_sae.pth")
            state_dict = torch.load(file_path, map_location=torch.device(device))
        except:
            continue
        sparse_ae = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            stddev_prior=run.config["stddev_prior"],
        )
        sparse_ae.load_state_dict(state_dict)
        sparse_ae.to(device)
        sae = SAEofSparseAutoencoder(sparse_ae)
        dash_filename = os.path.join(save_dir, f"{run.name}_vis.html")
        metrics_filename = os.path.join(save_dir, f"{run.name}_metrics.txt")
        if os.path.exists(dash_filename):
            continue

        activations_store = ActivationsStore.from_sae(
            model=model,
            sae=sae,
            streaming=True,
            store_batch_size_prompts=8,
            n_batches_in_buffer=8,
            device=device,
        )

        eval_metrics = run_evals(
            sae=sae,
            activation_store=activations_store,
            model=model,
            eval_config=EvalConfig(
                compute_kl=True,
                compute_ce_loss=True,
                compute_l2_norms=True,
                compute_sparsity_metrics=True,
                compute_variance_metrics=True,
            ),
        )

        # CE Loss score should be high for residual stream SAEs
        # ce loss without SAE should be fairly low < 3.5 suggesting the Model is being run correctly
        # ce loss with SAE shouldn't be massively higher
        print(f"saving metrics to {metrics_filename}")
        # print(run.name, eval_metrics)
        with open(metrics_filename, "w") as f:
            f.write(str(eval_metrics))

        feature_vis_config_gpt = sae_vis_runner.SaeVisConfig(
            hook_point=hook_point,
            features=list(range(25)),
            minibatch_size_features=2,
            minibatch_size_tokens=64,  # this is really prompt with the number of tokens determined by the sequence length
            verbose=False,
            device=device,
            cache_dir=Path(
                "demo_activations_cache"
            ),  # TODO: this will enable us to skip running the model for subsequent features.
            dtype="bfloat16",
        )

        runner = sae_vis_runner.SaeVisRunner(feature_vis_config_gpt)

        data = runner.run(
            encoder=sae,
            model=model,
            tokens=all_tokens,
        )

        print(f"saving dashboard to {dash_filename}")
        save_feature_centric_vis(sae_vis_data=data, filename=dash_filename)
