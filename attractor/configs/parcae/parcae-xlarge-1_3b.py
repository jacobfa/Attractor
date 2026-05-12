"""XLarge 1.3B Parcae model configuration with diagonal injection."""

from attractor.registry import DEFAULT_HF_ORG

configs = [
    # XLarge Parcae config ~1.3B params with diagonal injection
    dict(
        name="parcae-xlarge-1_3b",
        hf_config=dict(org=DEFAULT_HF_ORG, name="parcae-xlarge-1_3b"),
        # Context
        block_size=2048,
        vocab_size=32768,
        padding_multiple=64,
        n_embd=1536,
        num_attention_heads=12,
        num_key_value_heads=12,
        intermediate_size=6144,  # 4x n_embd
        bias=False,
        tie_embeddings=True,
        prelude_norm=True,
        # Model class
        architecture_class_name="Parcae",
        block_class_name="TransformerPreNormBlock",
        norm_class_name="RMSNorm",
        norm_eps=1e-5,
        mlp_class_name="BaseMLP",
        nonlin_name="ReLU2",
        qk_norm=True,
        logit_softcap=None,
        # Initialization
        init_strategy="scaled-zero",
        init_orthogonal=True,
        # Recurrent settings - diagonal injection
        injection_type="diagonal",
        state_init="like-init",
        recurrent_embedding_dimension=1536,
        recurrent_intermediation_embedding_dimension=6144,
        n_layers_in_recurrent_block=8,
        n_layers_in_prelude=8,
        n_layers_in_coda=8,
        mean_recurrence=8,
        sampling_scheme="poisson-truncated-full",
        mean_backprop_depth=4,
        recurrent_iteration_method="per-sequence",
    ),
]

