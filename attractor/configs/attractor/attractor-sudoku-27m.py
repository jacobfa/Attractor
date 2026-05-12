"""EQLM config for the Sudoku-Extreme task at HRM scale (~27M params).

Same width as eqlm-sudoku-7m (hidden=512, intermediate=2048, 8 heads,
bidirectional, vocab=11) but with 4 prelude blocks + 4 fixed-point blocks
(8 attention-bearing blocks total). This matches the HRM 27M parameter
budget on Sudoku-Extreme so EQLM can be benchmarked head-to-head with HRM.

Param accounting (hidden=512, intermediate=2304, BaseMLP, RMSNorm):
    per block        ~ 3.41M  (1.05M attn + 2.36M MLP + ~1k norms)
    8 blocks         ~ 27.3M
    + wte (11x512 padded to 16x512), ln_f, lm_head (tied) = ~27M total
"""
from attractor.registry import DEFAULT_HF_ORG


configs = [
    dict(
        name="attractor-sudoku-27m",
        hf_config=dict(org=DEFAULT_HF_ORG, name="attractor-sudoku-27m"),
        block_size=96,
        vocab_size=11,
        padding_multiple=16,
        n_embd=512,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=2304,
        bias=False,
        tie_embeddings=True,
        architecture_class_name="Attractor",
        block_class_name="TransformerPreNormBlock",
        norm_class_name="RMSNorm",
        norm_eps=1e-5,
        mlp_class_name="BaseMLP",
        nonlin_name="ReLU2",
        qk_norm=True,
        causal=False,
        logit_softcap=None,
        n_backbone_layers=4,
        n_fp_blocks=4,
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
