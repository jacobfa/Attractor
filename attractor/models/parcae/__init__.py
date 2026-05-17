from attractor.models.parcae.config import ParcaeConfig
from attractor.models.parcae.init import ParcaeInit


def __getattr__(name):
    if name == "Parcae":
        from attractor.models.parcae.parcae import Parcae
        return Parcae
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ParcaeConfig", "ParcaeInit", "Parcae"]
