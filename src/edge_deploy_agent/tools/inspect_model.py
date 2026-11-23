"""Static inspection of a Keras model: size, parameters and activation memory."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import keras

FLOAT32_BYTES = 4
# Layers that only reinterpret a buffer's shape; TFLite Micro runs them in place.
VIEW_LAYERS = {"Flatten", "Reshape", "InputLayer"}


def load_model(model_path: str | Path) -> keras.Model:
    return keras.models.load_model(model_path, compile=False)


def _elements(shape: tuple) -> int:
    """Number of elements in one sample (batch dimension dropped)."""
    dims = [d for d in shape[1:] if d is not None]
    return math.prod(dims) if dims else 1


def inspect_model(model_path: str | Path) -> dict[str, Any]:
    """Describe a Keras model so an agent can reason about deployment options.

    The peak activation estimate assumes a sequential graph where only one
    layer's input and output buffers are alive at the same time, which is how
    TFLite Micro reuses its tensor arena for simple feed-forward models.
    """
    model = load_model(model_path)

    layers = []
    peak_activation_elems = 0
    for layer in model.layers:
        in_shape = tuple(layer.input.shape) if hasattr(layer, "input") else ()
        out_shape = tuple(layer.output.shape)
        if layer.__class__.__name__ not in VIEW_LAYERS:
            live = (_elements(in_shape) if in_shape else 0) + _elements(out_shape)
            peak_activation_elems = max(peak_activation_elems, live)
        layers.append(
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": [d for d in out_shape[1:]],
                "params": layer.count_params(),
            }
        )

    params = model.count_params()
    return {
        "model_path": str(model_path),
        "input_shape": [d for d in model.input_shape[1:]],
        "output_shape": [d for d in model.output_shape[1:]],
        "num_layers": len(layers),
        "params": params,
        "float32_weights_bytes": params * FLOAT32_BYTES,
        "peak_activation_elements": peak_activation_elems,
        "layers": layers,
    }
