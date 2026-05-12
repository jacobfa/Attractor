"""EQLM config for the Sudoku-Extreme task (TRM benchmark).

Sized to ~7M parameters, matching the TRM-Att variant from
"Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871).

Differences from EQLM-LM configs:
- causal=False             # bidirectional attention (sudoku is structured-output)
- vocab_size=11            # PAD + digits 0..9 (matches sapientinc/sudoku-extreme)
- block_size=96            # >= 81 (the seq_len of a 9x9 grid)
- padding_multiple=16      # padded_vocab_size=16
- 1 prelude block + 1 fp block  # total ~7M params at hidden=512, expansion=4
"""
from attractor.registry import DEFAULT_HF_ORG


configs = [
    dict(
        name="eqlm-sudoku-7m",
        hf_config=dict(org=DEFAULT_HF_ORG, name="eqlm-sudoku-7m"),
        block_size=96,
        vocab_size=11,
        padding_multiple=16,
        n_embd=512,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=2048,
        bias=False,
        tie_embeddings=True,
        architecture_class_name="EQLM",
        block_class_name="TransformerPreNormBlock",
        norm_class_name="RMSNorm",
        norm_eps=1e-5,
        mlp_class_name="BaseMLP",
        nonlin_name="ReLU2",
        qk_norm=True,
        causal=False,
        logit_softcap=None,
        n_backbone_layers=1,
        n_fp_blocks=1,
        solver="anderson",
        max_iter=12,
        min_iter=4,
        tol=3e-4,
        anderson_m=5,
        anderson_beta=1.0,
        backward_type="onestep",
        backward_max_iter=12,
        backward_min_iter=4,
        backward_tol=3e-4,
        adjoint_grad_clip=1.0,
        layer_scale_init=0.5,
        gamma_max=0.75,
        fp_lr_scale=0.5,
        fp_wd=0.0,
        init_strategy="scaled-zero",
        init_orthogonal=True,
        rope_settings=dict(use_rope=True, rope_condense_ratio=1, rope_base=10_000),
    ),
]
