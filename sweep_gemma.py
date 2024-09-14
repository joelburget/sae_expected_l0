from generic_train import sweep, SweepConfig


config = SweepConfig(
    model_name="gemma-2-2b",
    # "Gemma 1's training data is not public. Use https://huggingface.co/datasets/monology/pile-uncopyrighted for a reasonable approximation."
    # https://opensourcemechanistic.slack.com/archives/C04T79RAW8Z/p1725371887228009?thread_ts=1725371842.001579&cid=C04T79RAW8Z
    dataset_name="monology/pile-uncopyrighted",
    dataset_is_tokenized=False,
    hook_point="blocks.6.hook_resid_post",
    hook_layer=5,
    config_path="gemma-config.yaml",
    expansion_factor=8,
    training_tokens=8_000_000,
)


if __name__ == "__main__":
    sweep(config)
