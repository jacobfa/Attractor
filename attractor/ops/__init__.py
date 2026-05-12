"""Optimized operations module."""

try:
    from attractor.ops.linear_cross_entropy import LinearCrossEntropyLoss, linear_cross_entropy
except (ImportError, NameError):
    LinearCrossEntropyLoss = None
    linear_cross_entropy = None

__all__ = ["LinearCrossEntropyLoss", "linear_cross_entropy"]
