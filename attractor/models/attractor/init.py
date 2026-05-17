from typing import Callable, Optional

from attractor.models.gpt.init import GPTInit


class AttractorInit(GPTInit):
    """Weight initialization for the Attractor model. Inherits all GPT-style
    init strategies; exists as a separate class so checkpoints and optimizer
    logs can identify attractor parameters distinctly."""

    def _get_layer_init(self, name_of_layer: str, layer_idx: int,
                        init_table: dict) -> Optional[Callable]:
        return super()._get_layer_init(name_of_layer, layer_idx, init_table)
