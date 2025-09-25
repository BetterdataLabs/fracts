"""Static data generator."""
from typing import Literal

from .base import StaticGenerator
from .ctgan import CTGANGenerator, TVAEGenerator
from .rtf import RTFTabGenerator, RTFRelGenerator
from .arf import ARFGenerator

static_generators = {
    "ctgan": CTGANGenerator, "tvae": TVAEGenerator, "rtf-t": RTFTabGenerator, "rtf-r": RTFRelGenerator, "arf": ARFGenerator
}


def create_static_generator(
        model_type: Literal["ctgan", "tvae", "rtf-t", "rtf-r", "arf"] = "ctgan", **kwargs
) -> StaticGenerator:
    return static_generators[model_type](**kwargs)
