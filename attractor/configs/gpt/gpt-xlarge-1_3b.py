from attractor.registry import DEFAULT_HF_ORG


configs = [
    dict(
        name="gpt-xlarge-1_3b",
        hf_config=dict(org=DEFAULT_HF_ORG, name="gpt-xlarge-1_6b"),
        # Context
        block_size=2048,
        vocab_size=32768,
        padding_multiple=64,
        n_embd=1536,
        n_layer=24,
        num_attention_heads=12,
        num_key_value_heads=12,
        intermediate_size=6144,  # 4x n_embd
        bias=False,
        tie_embeddings=True,
        # Model class
        architecture_class_name="GPT",
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
    ),
]

