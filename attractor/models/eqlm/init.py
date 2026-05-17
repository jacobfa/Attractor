from typing import Callable, Optional

from attractor.models.gpt.init import GPTInit


class EQLMInit(GPTInit):

    def _get_layer_init(self, name_of_layer: str, layer_idx: int,
                        init_table: dict) -> Optional[Callable]:
        return super()._get_layer_init(name_of_layer, layer_idx, init_table)
