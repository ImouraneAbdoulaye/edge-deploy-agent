"""Measure the accuracy of a TFLite model and check it against a hardware target."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


def evaluate_tflite(
    tflite_path: str | Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_samples: int = 1000,
) -> dict[str, Any]:
    """Run the TFLite interpreter on the test set and return top-1 accuracy."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    in_scale, in_zero = inp["quantization"]
    is_quantized_input = inp["dtype"] in (np.int8, np.uint8)

    n = min(max_samples, len(x_test))
    correct = 0
    for x, y in zip(x_test[:n], y_test[:n]):
        sample = x[None, ...]
        if is_quantized_input:
            sample = np.round(sample / in_scale + in_zero).astype(inp["dtype"])
        else:
            sample = sample.astype(inp["dtype"])
        interpreter.set_tensor(inp["index"], sample)
        interpreter.invoke()
        logits = interpreter.get_tensor(out["index"])[0]
        correct += int(np.argmax(logits) == y)

    return {"tflite_path": str(tflite_path), "accuracy": correct / n, "num_samples": n}


def check_constraints(
    flash_bytes: int,
    est_ram_bytes: int,
    accuracy: float,
    target: dict[str, Any],
    reference_accuracy: float,
    max_accuracy_drop: float = 0.01,
) -> dict[str, Any]:
    """Compare measured metrics with a target's memory budget and an accuracy tolerance.

    ``target`` holds ``flash_budget_kb`` and ``ram_budget_kb``: what is left for the
    model after firmware/OS, not the raw chip memory. Accuracy may drop by at most
    ``max_accuracy_drop`` (absolute) below ``reference_accuracy``, the float32 model's.
    """
    flash_budget = target["flash_budget_kb"] * 1024
    ram_budget = target["ram_budget_kb"] * 1024
    min_accuracy = reference_accuracy - max_accuracy_drop
    violations = []
    if flash_bytes > flash_budget:
        violations.append(f"flash {flash_bytes / 1024:.1f} KB > budget {target['flash_budget_kb']} KB")
    if est_ram_bytes > ram_budget:
        violations.append(f"RAM {est_ram_bytes / 1024:.1f} KB > budget {target['ram_budget_kb']} KB")
    if accuracy < min_accuracy:
        violations.append(
            f"accuracy {accuracy:.3f} < {min_accuracy:.3f} (reference {reference_accuracy:.3f} "
            f"- tolerance {max_accuracy_drop})"
        )
    return {
        "fits": not violations,
        "violations": violations,
        "flash_usage_pct": round(100 * flash_bytes / flash_budget, 1),
        "ram_usage_pct": round(100 * est_ram_bytes / ram_budget, 1),
    }
