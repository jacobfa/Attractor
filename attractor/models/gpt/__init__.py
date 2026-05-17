from attractor.models.gpt.config import GPTConfig
from attractor.models.gpt.init import GPTInit


def __getattr__(name):
    if name == "GPT":
        from attractor.models.gpt.gpt import GPT
        return GPT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GPTConfig", "GPT", "GPTInit"]
