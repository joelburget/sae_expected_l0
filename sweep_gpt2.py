from generic_train import sweep, SweepConfig

config = SweepConfig(
    model_name="gpt2",
    dataset_name="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
    hook_point="blocks.6.hook_resid_post",
    hook_layer=6,
    config_path="gpt2-config.yaml",
    expansion_factor=8,
    training_tokens=8_000_000,
)

if __name__ == "__main__":
    sweep(config)
