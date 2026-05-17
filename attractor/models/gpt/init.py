# GPT-specific initialization utilities
"""Initialization for GPT/Transformer models.

Handles transformer-specific modules:
- QKV projections (fused and separate)
- GLU/MLP layers
- Attention output projections
"""

import torch
from typing import Optional, Callable

from attractor.utils.init import Init


@torch.no_grad()
def init_qkv(qkv_tensor, init_fn, qk_std, v_std, dim, head_dim):
    """Initialize fused QKV projection tensor."""
    s = qkv_tensor.shape[0]
    n_kv_heads = (s - dim) // (2 * head_dim)
    shapes = [dim, n_kv_heads * head_dim, n_kv_heads * head_dim]

    Q, K, V = (
        qkv_tensor.new_empty([shapes[0], dim]),
        qkv_tensor.new_empty([shapes[1], dim]),
        qkv_tensor.new_empty([shapes[2], dim]),
    )
    init_fn(Q, qk_std)
    init_fn(K, qk_std)
    init_fn(V, v_std)
    qkv_tensor.data.copy_(torch.cat([Q, K, V], dim=0).contiguous())


@torch.no_grad()
def init_qkv_gate(qkv_gate_tensor, init_fn, qk_std, v_std, dim, head_dim, gate_dim):
    """Initialize Q+gate+K+V tensor. Gate initialized to zeros for sigmoid(0)=0.5."""
    s = qkv_gate_tensor.shape[0]
    kv_dim = (s - dim - gate_dim) // 2

    Q = qkv_gate_tensor.new_empty([dim, dim])
    G = qkv_gate_tensor.new_zeros([gate_dim, dim])
    K = qkv_gate_tensor.new_empty([kv_dim, dim])
    V = qkv_gate_tensor.new_empty([kv_dim, dim])

    init_fn(Q, qk_std)
    init_fn(K, qk_std)
    init_fn(V, v_std)

    qkv_gate_tensor.data.copy_(torch.cat([Q, G, K, V], dim=0).contiguous())


@torch.no_grad()
def init_qkv_gate_diagonal(qkv_gate_tensor, init_fn, qk_std, v_std, dim, head_dim, gate_dim):
    """Initialize Q+gate+K+V tensor with diagonal Q/K. Gate initialized to zeros."""
    s = qkv_gate_tensor.shape[0]
    kv_dim = (s - dim - gate_dim) // 2

    Q = torch.eye(dim, dtype=qkv_gate_tensor.dtype, device=qkv_gate_tensor.device) * qk_std
    G = qkv_gate_tensor.new_zeros([gate_dim, dim])
    K = torch.eye(kv_dim, dim, dtype=qkv_gate_tensor.dtype, device=qkv_gate_tensor.device) * qk_std
    V = qkv_gate_tensor.new_empty([kv_dim, dim])

    init_fn(V, v_std)

    qkv_gate_tensor.data.copy_(torch.cat([Q, G, K, V], dim=0).contiguous())


@torch.no_grad()
def init_qk_diagonal(qkv_tensor, init_fn, qk_std, v_std, dim, head_dim):
    """Initialize QKV with diagonal Q and K matrices."""
    s = qkv_tensor.shape[0]
    n_kv_heads = (s - dim) // (2 * head_dim)
    shapes = [dim, n_kv_heads * head_dim, n_kv_heads * head_dim]
    assert n_kv_heads == dim // head_dim

    Q = torch.eye(dim, dtype=qkv_tensor.dtype, device=qkv_tensor.device) * qk_std
    K = torch.eye(dim, dtype=qkv_tensor.dtype, device=qkv_tensor.device) * qk_std
    V = qkv_tensor.new_empty([shapes[2], dim])
    init_fn(V, v_std)
    qkv_tensor.data.copy_(torch.cat([Q, K, V], dim=0).contiguous())


@torch.no_grad()
def init_glu(glu_tensor, init_fn, w1_std, w2_std):
    """Initialize GLU (gated linear unit) tensor."""
    g, h = glu_tensor.shape
    W1, W2 = (
        glu_tensor.new_empty([g // 2, h]),
        glu_tensor.new_empty([g // 2, h]),
    )
    init_fn(W1, w1_std)
    init_fn(W2, w2_std)
    glu_tensor.data.copy_(torch.cat([W1, W2], dim=0).contiguous())


class GPTInit(Init):
    """GPT/Transformer-specific initialization.
    
    Handles:
    - qkv: Fused query/key/value projections
    - qkv-gate: QKV with gating mechanism
    - qkv-diagonal: QKV with diagonal Q/K initialization
    - glu: Gated linear units for MLP
    """

    def _get_layer_init(self, name_of_layer: str, layer_idx: int, init_table: dict) -> Optional[Callable]:
        """Handle GPT-specific layer initialization."""
        mu = self.mup_model_scaling_factor

        # QKV with gating
        if "qkv-gate" in name_of_layer:
            qk_std = next((init_table.get(key) for key in ["q", "in_proj", "std"] if key in init_table), None)
            v_std = next((init_table.get(key) for key in ["v", "in_proj", "std"] if key in init_table), None)
            if qk_std is None or v_std is None:
                raise ValueError(f"Could not resolve init of layer {name_of_layer}")
            qk_std /= mu
            v_std /= mu

            def _infer_gate_dim(tensor):
                # gate_dim = total - q_dim - 2*kv_dim = total - 3*dim (for standard MHA)
                s = tensor.shape[0]
                gate_dim = s - 3 * self.dim
                return gate_dim

            if "diagonal" not in name_of_layer:
                def init(tensor, qk_std=qk_std, v_std=v_std):
                    gate_dim = _infer_gate_dim(tensor)
                    if self.verbose:
                        print(f"Init layer {layer_idx} {name_of_layer} with qk_std={qk_std:2.4f}, v_std={v_std:2.4f}, gate_dim={gate_dim}.")
                    init_qkv_gate(tensor, self.normal_, float(qk_std), float(v_std), self.dim, self.head_dim, gate_dim)
            else:
                def init(tensor, qk_std=qk_std, v_std=v_std):
                    gate_dim = _infer_gate_dim(tensor)
                    if self.verbose:
                        print(f"Init layer {layer_idx} {name_of_layer} diag qk_std={qk_std:2.4f}, v_std={v_std:2.4f}, gate_dim={gate_dim}.")
                    init_qkv_gate_diagonal(tensor, self.normal_, float(qk_std), float(v_std), self.dim, self.head_dim, gate_dim)

            return init

        # Standard QKV
        elif "qkv" in name_of_layer:
            qk_std = next((init_table.get(key) for key in ["q", "in_proj", "std"] if key in init_table), None)
            v_std = next((init_table.get(key) for key in ["v", "in_proj", "std"] if key in init_table), None)
            if qk_std is None or v_std is None:
                raise ValueError(f"Could not resolve init of layer {name_of_layer}")
            qk_std /= mu
            v_std /= mu

            if "diagonal" not in name_of_layer:
                def init(tensor):
                    if self.verbose:
                        print(f"Init layer {layer_idx} {name_of_layer} with qk_std={qk_std:2.4f}, v_std={v_std:2.4f}.")
                    init_qkv(tensor, self.normal_, float(qk_std), float(v_std), self.dim, self.head_dim)
            else:
                def init(tensor):
                    if self.verbose:
                        print(f"Init layer {layer_idx} {name_of_layer} diag s={qk_std:2.4f}, v_std={v_std:2.4f}.")
                    init_qk_diagonal(tensor, self.normal_, float(qk_std), float(v_std), self.dim, self.head_dim)

            return init

        # GLU (Gated Linear Unit)
        elif "glu" in name_of_layer:
            w1_std = next((init_table.get(key) for key in ["w1", "mlp", "in_proj", "std"] if key in init_table), None)
            w2_std = next((init_table.get(key) for key in ["w2", "mlp", "out_proj", "std"] if key in init_table), None)
            if w1_std is None or w2_std is None:
                raise ValueError(f"Could not resolve init of layer {name_of_layer}")
            w1_std /= mu
            w2_std /= mu

            def init(tensor):
                if self.verbose:
                    print(f"Init layer {layer_idx} {name_of_layer} with w1_std={w1_std:2.4f}, w2_std={w2_std:2.4f}.")
                init_glu(tensor, self.normal_, float(w1_std), float(w1_std))

            return init

        # Not handled by GPT-specific init
        return None



