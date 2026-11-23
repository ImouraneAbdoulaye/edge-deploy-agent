"""Qualia-CodeGen backend: PyTorch model -> portable C library (float32, int16 or int8).

Qualia-CodeGen (LEAT, Université Côte d'Azur) generates plain C inference code with
no runtime dependency, an alternative to TFLite Micro's interpreter. Fixed-point
quantization uses one power-of-two scale factor per layer, computed here from
calibration data (post-training quantization).

Memory figures come from the compiled object file: the size of the weight arrays
(flash) and of the static activation buffers (RAM) do not depend on the CPU the
code is compiled for, so measuring them on the host gives the MCU footprint of the
model itself (inference code excluded).
"""

from __future__ import annotations

import contextlib
import io
import math
import shutil
import subprocess
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import fx, nn

from edge_deploy_agent.torch_models import TORCH_MODELS

QUALIA_MODES: dict[str, dict[str, Any]] = {
    "float32": {"number_type": float, "width": 32, "long_width": 32,
                "description": "No quantization (reference)."},
    "int16": {"number_type": int, "width": 16, "long_width": 32,
              "description": "16-bit fixed point, 32-bit accumulators. ~2x smaller, near-float accuracy."},
    "int8": {"number_type": int, "width": 8, "long_width": 16,
             "description": "8-bit fixed point, 16-bit accumulators. ~4x smaller and less RAM; "
                            "power-of-two per-layer scales can cost accuracy."},
}  # fmt: skip


def load_torch_model(model_path: str | Path, input_shape: tuple[int, ...] | None = None) -> nn.Module:
    """Load a PyTorch model for Qualia-CodeGen.

    Reference models are stored as state_dicts and rebuilt from ``TORCH_MODELS``. Any
    other ``.pt`` must be a full model saved with ``torch.save(model, path)``; it is
    unpickled, so only load files you trust. Qualia needs ``model.input_shape``
    (channels last, no batch): it is set from ``input_shape`` when missing.
    """
    stem = Path(model_path).stem
    if stem in TORCH_MODELS:
        model = TORCH_MODELS[stem]()
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        model = torch.load(model_path, weights_only=False)
        if not isinstance(model, nn.Module):
            raise TypeError(f"{model_path} must contain a full nn.Module saved with torch.save(model, path)")
    if not hasattr(model, "input_shape") and input_shape is not None:
        model.input_shape = tuple(input_shape)
    return model.eval()


def channels_first(x: np.ndarray) -> np.ndarray:
    """(N, ..., C) channels-last batch -> (N, C, ...) for PyTorch."""
    return np.moveaxis(x, -1, 1).astype(np.float32)


def _fractional_bits(max_abs: float, width: int) -> int:
    """Fractional bits for a symmetric fixed-point tensor whose largest magnitude is ``max_abs``.

    Same rule as qualia-core's ``Quantizer.fractional_bits`` (minmax range setting):
    ``width - 1 - (floor(log2(max_abs)) + 1)``, and ``width - 2`` for an all-zero tensor.
    """
    if max_abs == 0:
        return width - 2
    return width - 1 - (math.floor(math.log2(max_abs)) + 1)


class _RangeRecorder(fx.Interpreter):
    """Runs the traced graph and records the max |value| of each node's input and output."""

    def __init__(self, gm: fx.GraphModule) -> None:
        super().__init__(gm)
        self.max_in: dict[str, float] = {}
        self.max_out: dict[str, float] = {}

    def run_node(self, n: fx.Node) -> Any:
        out = super().run_node(n)
        if n.op in ("call_module", "call_function", "call_method") and isinstance(out, torch.Tensor):
            args = [a for a in n.args if isinstance(a, fx.Node)]
            if args:
                x = self.env[args[0]]
                self.max_in[n.name] = max(self.max_in.get(n.name, 0.0), float(x.abs().max()))
            self.max_out[n.name] = max(self.max_out.get(n.name, 0.0), float(out.abs().max()))
        return out


def compute_activations_range(
    model: nn.Module,
    x_calib: np.ndarray,
    width: int,
    long_width: int,
    out_file: Path,
    accumulator_headroom: bool = True,
    batch_size: int = 500,
) -> Path:
    """Write the per-layer quantization file that Qualia-CodeGen expects.

    One line per graph node: name, input/activation/weights/bias bits, the matching
    power-of-two scale factors (fractional bits) and the round modes.

    Follows qualia-core's post-training defaults: minmax range setting, activation
    ranges as the running max over the calibration data, and weights and biases
    sharing one scale factor (qualia-core concatenates them when no separate bias
    quantizer is configured).

    Addition over qualia-core (``accumulator_headroom``): products are accumulated at
    scale 2**(input_q + weights_q) in a ``long_width`` integer. Without
    quantization-aware training, max-precision weights can overflow it (observed on
    int16 / 32-bit accumulators), so weights precision is lowered when needed to
    leave room for the largest observed sum, with one bit of margin for partial sums.
    """
    gm = fx.symbolic_trace(model)
    modules = dict(gm.named_modules())
    recorder = _RangeRecorder(gm)
    with torch.no_grad():
        for i in range(0, len(x_calib), batch_size):
            recorder.run(torch.from_numpy(x_calib[i : i + batch_size]))

    lines = []
    act_q_of: dict[str, int] = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            act_q_of[node.name] = _fractional_bits(float(np.abs(x_calib).max()), width)
            continue
        if node.name not in recorder.max_out:
            continue
        module = modules.get(node.target) if node.op == "call_module" else None
        w = getattr(module, "weight", None)
        b = getattr(module, "bias", None)
        prev = next(a for a in node.args if isinstance(a, fx.Node))
        input_q = act_q_of[prev.name]
        act_q = _fractional_bits(recorder.max_out[node.name], width)

        weights_q, bias_q = 0, None
        if w is not None:
            max_wb = max(float(w.detach().abs().max()), float(b.detach().abs().max()) if b is not None else 0.0)
            weights_q = _fractional_bits(max_wb, width)
            if accumulator_headroom:
                acc_bits = math.floor(math.log2(2 * recorder.max_out[node.name] + 1e-12)) + 1
                weights_q = min(weights_q, long_width - 1 - acc_bits - input_q)
            if b is not None:
                bias_q = weights_q
        act_q_of[node.name] = act_q
        lines.append(
            f"{node.name},{width},{width},{width},{width if b is not None else 'None'},"
            f"{input_q},{act_q},{weights_q},{bias_q},NEAREST,NEAREST,NEAREST"
        )
    out_file.write_text("\n".join(lines) + "\n")
    return out_file


def _symbol_sizes(object_file: Path) -> dict[str, int]:
    """Map symbol name -> size in bytes, from ``nm``."""
    out = subprocess.run(
        ["nm", "-S", "--size-sort", str(object_file)], check=True, capture_output=True, text=True
    ).stdout
    sizes = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4:
            sizes[parts[3]] = int(parts[1], 16)
    return sizes


def measure_memory(c_dir: Path) -> dict[str, int]:
    """Compile the generated model and sum the weight (flash) and activation (RAM) buffers."""
    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / "model.o"
        subprocess.run(
            ["gcc", "-c", "-O2", "-include", str(c_dir / "include" / "defines.h"),
             f"-I{c_dir}", f"-I{c_dir / 'include'}", str(c_dir / "model.c"), "-o", str(obj)],
            check=True, capture_output=True, text=True,
        )  # fmt: skip
        sizes = _symbol_sizes(obj)
    ram = sum(s for name, s in sizes.items() if "activations" in name)
    flash = sum(s for name, s in sizes.items() if name.endswith(("_kernel", "_bias")))
    return {"flash_bytes": flash, "ram_bytes": ram}


def convert_qualia(
    model_path: str | Path,
    mode: str,
    out_dir: str | Path,
    x_calib: np.ndarray | None = None,
    calib_samples: int = 2000,
    accumulator_headroom: bool = True,
) -> dict[str, Any]:
    """Generate the C library for ``model_path`` in ``mode`` and measure its footprint.

    ``x_calib`` is channels last, (N, H, W, C) or (N, L, C), like the rest of the project.
    """
    from qualia_codegen_core import Converter
    from qualia_codegen_core.graph.ActivationsRange import ActivationsRange
    from qualia_codegen_core.graph.TorchModelGraph import TorchModelGraph
    from qualia_codegen_core.main import annotate_quantization
    from qualia_codegen_core.MetricsConverter import MetricsConverter

    if mode not in QUALIA_MODES:
        raise ValueError(f"Unknown mode {mode!r}. Available: {sorted(QUALIA_MODES)}")
    cfg = QUALIA_MODES[mode]
    if cfg["number_type"] is int and x_calib is None:
        raise ValueError(f"{mode} needs calibration data (x_calib).")

    model = load_torch_model(model_path, input_shape=None if x_calib is None else x_calib.shape[1:])
    c_dir = Path(out_dir) / f"{Path(model_path).stem}_{mode}"
    shutil.rmtree(c_dir, ignore_errors=True)
    c_dir.mkdir(parents=True)

    with contextlib.redirect_stdout(io.StringIO()):  # Qualia prints the traced graph
        graph = TorchModelGraph(model).convert()
    if not graph:
        raise RuntimeError("Qualia-CodeGen could not trace the model.")

    ranges = ActivationsRange()
    if cfg["number_type"] is int:
        calib = channels_first(x_calib[:calib_samples])
        range_file = compute_activations_range(
            model, calib, cfg["width"], cfg["long_width"], c_dir / "activations_range.txt",
            accumulator_headroom=accumulator_headroom,
        )
        ranges = ranges.load(range_file, input_layer_name=graph.nodes[0].layer.name)

    if not annotate_quantization(graph, ranges, cfg["number_type"], cfg["width"], cfg["long_width"]):
        raise RuntimeError("Qualia-CodeGen quantization annotation failed.")
    if Converter(output_path=c_dir).convert_model(graph) is False:
        raise RuntimeError("Qualia-CodeGen code generation failed.")
    MetricsConverter(output_path=c_dir).convert_metrics(["acc"])

    return {"mode": mode, "backend": "qualia", "c_dir": str(c_dir), **measure_memory(c_dir)}


def _qualia_linux_example() -> Path:
    return Path(str(files("qualia_codegen_core") / "examples" / "Linux"))


def evaluate_qualia(
    c_dir: str | Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_samples: int = 1000,
) -> dict[str, Any]:
    """Build the generated C code for the host (Qualia's Linux demo) and measure accuracy.

    The C code is channels-last, matching the (N, H, W, C) layout of ``x_test``.
    """
    c_dir = Path(c_dir)
    build = c_dir / "build"
    subprocess.run(
        ["cmake", "-S", str(_qualia_linux_example()), "-B", str(build), "-G", "Ninja",
         f"-DMODEL_DIR={c_dir.resolve()}", "-DCMAKE_BUILD_TYPE=Release"],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    subprocess.run(["cmake", "--build", str(build), "--target", "Linux"], check=True, capture_output=True, text=True)

    n = min(max_samples, len(x_test))
    x = x_test[:n].reshape(n, -1)
    num_classes = int(y_test.max()) + 1
    y = np.eye(num_classes, dtype=np.float32)[y_test[:n]]
    np.savetxt(build / "x.csv", x, delimiter=",", fmt="%.6f")
    np.savetxt(build / "y.csv", y, delimiter=",", fmt="%d")

    run = subprocess.run(
        [str(build / "Linux"), str(build / "x.csv"), str(build / "y.csv")],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    # The demo prints "<Metric>=<value>" on stderr.
    for line in run.stderr.splitlines():
        if line.lower().startswith("acc"):
            return {"c_dir": str(c_dir), "accuracy": float(line.split("=")[1]), "num_samples": n}
    raise RuntimeError(f"No accuracy in Qualia demo output:\n{run.stderr}")
