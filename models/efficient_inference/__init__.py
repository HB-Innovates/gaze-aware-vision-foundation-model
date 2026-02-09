"""Efficient inference modules for power-optimized deployment."""

from .snn_converter import SNNConverter, convert_to_snn
from .quantization import quantize_model

__all__ = [
    "SNNConverter",
    "convert_to_snn",
    "quantize_model",
]
