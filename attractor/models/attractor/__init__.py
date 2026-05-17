from attractor.models.attractor.config import AttractorConfig
from attractor.models.attractor.init import AttractorInit


def __getattr__(name):
    if name == "Attractor":
        from attractor.models.attractor.attractor import Attractor
        return Attractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AttractorConfig", "AttractorInit", "Attractor"]
