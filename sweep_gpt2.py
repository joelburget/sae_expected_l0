from generic_train import sweep, SweepConfig

config = SweepConfig(
    model_name="gpt2",
    dataset_name="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
    dataset_is_tokenized=True,
    hook_point="blocks.5.hook_resid_post",
    hook_layer=5,
    config_path="gpt2-config.yaml",
    expansion_factor=8,
    training_tokens=8_000_000,
)

if __name__ == "__main__":
    sweep(config)
