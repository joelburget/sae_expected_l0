from generic_train import sweep, SweepConfig

if __name__ == "__main__":
    for layer in range(3):
        for training_tokens in [1_000_000, 2_000_000]:
            for expansion_factor in [8, 16]:
                config = SweepConfig(
                    model_name="gpt2",
                    dataset_name="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
                    dataset_is_tokenized=True,
                    hook_point=f"blocks.{layer}.hook_resid_post",
                    hook_layer=layer,
                    config_path="gpt2-config.yaml",
                    expansion_factor=expansion_factor,
                    training_tokens=training_tokens,
                )
                sweep(config)

    for layer in range(3):
        for training_tokens in [1_000_000, 2_000_000]:
            for expansion_factor in [8, 16]:
                config = SweepConfig(
                    model_name="gemma-2-2b",
                    # "Gemma 1's training data is not public. Use https://huggingface.co/datasets/monology/pile-uncopyrighted for a reasonable approximation."
                    # https://opensourcemechanistic.slack.com/archives/C04T79RAW8Z/p1725371887228009?thread_ts=1725371842.001579&cid=C04T79RAW8Z
                    dataset_name="NeelNanda/pile-10k",
                    dataset_is_tokenized=False,
                    hook_point=f"blocks.{layer}.hook_resid_post",
                    hook_layer=layer,
                    config_path="gemma-config.yaml",
                    expansion_factor=expansion_factor,
                    training_tokens=training_tokens,
                )
                sweep(config)
