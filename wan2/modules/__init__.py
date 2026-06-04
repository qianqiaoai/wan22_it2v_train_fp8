# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from importlib import import_module

from .attention import flash_attention
from .model import WanModel
from .vae2_1 import Wan2_1_VAE
from .vae2_2 import Wan2_2_VAE

__all__ = [
    'Wan2_1_VAE',
    'Wan2_2_VAE',
    'WanModel',
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
    'HuggingfaceTokenizer',
    'flash_attention',
]


def __getattr__(name):
    if name in {'T5Model', 'T5Encoder', 'T5Decoder', 'T5EncoderModel'}:
        t5 = import_module('.t5', __name__)
        return getattr(t5, name)
    if name == 'HuggingfaceTokenizer':
        from .tokenizers import HuggingfaceTokenizer
        return HuggingfaceTokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
