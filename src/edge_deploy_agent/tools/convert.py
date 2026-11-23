"""Convert a Keras model to TFLite with different quantization strategies."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from edge_deploy_agent.tools.inspect_model import inspect_model, load_model

# Bytes per activation element at inference time for each mode.
# Only full integer quantization makes activations int8; the other modes keep
# float32 activations even when the weights are compressed.
QUANT_MODES: dict[str, dict[str, Any]] = {
    "float32": {"activation_bytes": 4, "description": "No quantization (reference)."},
    "float16": {"activation_bytes": 4, "description": "Weights stored as float16, ~2x smaller."},
    "dynamic_int8": {
        "activation_bytes": 4,
        "description": "Weights int8, activations float32. ~4x smaller, no calibration data.",
    },
    "full_int8": {
        "activation_bytes": 1,
        "description": "Weights and activations int8 (needs calibration data). "
        "~4x smaller and ~4x less RAM; required by most NPUs and int8-only MCU kernels.",
    },
}


def _representative_dataset(x_calib: np.ndarray, n: int) -> Callable[[], Iterator[list[np.ndarray]]]:
    def gen() -> Iterator[list[np.ndarray]]:
        for sample in x_calib[:n]:
            yield [sample[None, ...].astype(np.float32)]

    return gen


def convert_model(
    model_path: str | Path,
    mode: str,
    out_dir: str | Path,
    x_calib: np.ndarray | None = None,
    calib_samples: int = 200,
) -> dict[str, Any]:
    """Convert ``model_path`` to a ``.tflite`` file using quantization ``mode``.

    Returns the file path, its size (flash footprint of the weights + graph) and an
    estimate of the RAM needed for activations.
    """
    if mode not in QUANT_MODES:
        raise ValueError(f"Unknown mode {mode!r}. Available: {sorted(QUANT_MODES)}")
    if mode == "full_int8" and x_calib is None:
        raise ValueError("full_int8 needs calibration data (x_calib).")

    model = load_model(model_path)
    with tempfile.TemporaryDirectory() as saved_model_dir:
        # Keras 3 models go through a SavedModel export so variables get frozen.
        model.export(saved_model_dir, format="tf_saved_model", verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_bytes = _convert(converter, mode, x_calib, calib_samples)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(model_path).stem}_{mode}.tflite"
    out_path.write_bytes(tflite_bytes)

    info = inspect_model(model_path)
    est_ram = info["peak_activation_elements"] * QUANT_MODES[mode]["activation_bytes"]
    return {
        "mode": mode,
        "tflite_path": str(out_path),
        "flash_bytes": len(tflite_bytes),
        "est_ram_bytes": est_ram,
    }


def _convert(
    converter: tf.lite.TFLiteConverter, mode: str, x_calib: np.ndarray | None, calib_samples: int
) -> bytes:

    if mode == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif mode == "dynamic_int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif mode == "full_int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset(x_calib, calib_samples)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    return converter.convert()
