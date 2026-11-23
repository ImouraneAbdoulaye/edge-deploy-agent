"""Static inspection of a Keras model: size, parameters and activation memory."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import keras

FLOAT32_BYTES = 4
# Layers that only reinterpret a buffer's shape; TFLite Micro runs them in place.
VIEW_LAYERS = {"Flatten", "Reshape", "InputLayer"}
# PyTorch keeps activations as separate modules; deployed code applies them in place.
IN_PLACE_TORCH_LAYERS = {"Identity", "Dropout", "ReLU", "ReLU6", "Sigmoid", "Tanh"}


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


def inspect_torch_model(model: Any, input_shape: tuple[int, ...]) -> dict[str, Any]:
    """Same description as ``inspect_model`` for a PyTorch model (input shape channels last)."""
    import torch

    layers, live = [], []

    def hook(module: Any, inputs: tuple, output: Any) -> None:
        n_in = inputs[0][0].numel() if inputs and hasattr(inputs[0], "numel") else 0
        n_out = output[0].numel()
        if module.__class__.__name__ not in VIEW_LAYERS | IN_PLACE_TORCH_LAYERS:
            live.append(n_in + n_out)
        layers.append(
            {
                "name": module.__class__.__name__.lower(),
                "type": module.__class__.__name__,
                "output_shape": list(output.shape[1:]),
                "params": sum(p.numel() for p in module.parameters(recurse=False)),
            }
        )

    leaves = [m for m in model.modules() if not list(m.children())]
    handles = [m.register_forward_hook(hook) for m in leaves]
    channels_first_shape = (1, input_shape[-1], *input_shape[:-1])
    with torch.no_grad():
        model(torch.zeros(channels_first_shape))
    for h in handles:
        h.remove()

    params = sum(p.numel() for p in model.parameters())
    return {
        "input_shape": list(input_shape),
        "num_layers": len(layers),
        "params": params,
        "float32_weights_bytes": params * FLOAT32_BYTES,
        "peak_activation_elements": max(live, default=0),
        "layers": layers,
    }
