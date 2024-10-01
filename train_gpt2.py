from generic_train import train, TrainConfig

if __name__ == "__main__":
    for layer in range(12):
        for hook_point in ["hook_resid_pre", "hook_resid_post"]:
            config = TrainConfig(
                model_name="gpt2",
                dataset_name="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
                dataset_is_tokenized=True,
                hook_point=f"blocks.{layer}.{hook_point}",
                hook_layer=layer,
                config_path="gpt2-config.yaml",
                expansion_factor=8,
                training_tokens=8_000_000,
                stddev_prior=0.015,
                learning_rate=0.001,
                reconstruction_coefficient=200000000.0,
            )
            model, sae = train(config)
