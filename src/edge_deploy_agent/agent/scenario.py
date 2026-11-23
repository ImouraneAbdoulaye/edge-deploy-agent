"""A deployment scenario (model + target) and the tools the agent can call on it."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from edge_deploy_agent.tools.deploy import BACKENDS, ROOT, deploy
from edge_deploy_agent.tools.evaluate import check_constraints
from edge_deploy_agent.tools.inspect_model import inspect_model

REFERENCE_CACHE = ROOT / "artifacts" / "reference_accuracy.json"
DEFAULT_MAX_ACCURACY_DROP = 0.01


def load_targets() -> dict[str, dict[str, Any]]:
    return yaml.safe_load((ROOT / "configs" / "targets.yaml").read_text())


def reference_accuracy(model_name: str, backend: str) -> float:
    """Accuracy of the unquantized (float32) model on ``backend``, cached on disk."""
    cache = json.loads(REFERENCE_CACHE.read_text()) if REFERENCE_CACHE.exists() else {}
    key = f"{model_name}/{backend}"
    if key not in cache:
        res = deploy(model_name, backend, "float32", ROOT / "artifacts" / "reference" / backend)
        cache[key] = res["accuracy"]
        REFERENCE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE_CACHE.write_text(json.dumps(cache, indent=2))
    return cache[key]


@dataclass
class Scenario:
    model_name: str
    target_name: str
    max_accuracy_drop: float = DEFAULT_MAX_ACCURACY_DROP
    out_dir: Path = ROOT / "artifacts" / "agent"
    target: dict[str, Any] = field(init=False)
    reference: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.target = load_targets()[self.target_name]
        self.reference = {b: reference_accuracy(self.model_name, b) for b in BACKENDS}

    def brief(self) -> str:
        """Task description given to the agent as the first user message."""
        refs = ", ".join(f"{b}: {acc:.3f}" for b, acc in self.reference.items())
        return (
            f"Deploy the model `{self.model_name}` on the target `{self.target_name}` "
            f"({self.target['description']}).\n"
            f"- Flash budget for the model: {self.target['flash_budget_kb']} KB\n"
            f"- RAM budget for the model: {self.target['ram_budget_kb']} KB\n"
            f"- Max accuracy drop vs. the float32 model: {self.max_accuracy_drop:.3f} (absolute)\n"
            f"- Float32 reference accuracy per backend: {refs}\n"
            "Find a backend and mode that satisfies all three constraints, or show that none exists."
        )

    # ---- tools -----------------------------------------------------------------

    def inspect(self) -> dict[str, Any]:
        info = inspect_model(ROOT / "models" / f"{self.model_name}.keras")
        params, act = info["params"], info["peak_activation_elements"]
        return {
            "params": params,
            "layers": [f"{lyr['type']} {lyr['output_shape']} ({lyr['params']} params)" for lyr in info["layers"]],
            "weights_kb": {
                "32-bit": round(params * 4 / 1024, 1),
                "16-bit": round(params * 2 / 1024, 1),
                "8-bit": round(params / 1024, 1),
            },
            "peak_activation_elements": act,
            "peak_activation_kb": {
                "32-bit": round(act * 4 / 1024, 1),
                "16-bit": round(act * 2 / 1024, 1),
                "8-bit": round(act / 1024, 1),
            },
            "note": "Estimates. Real flash adds a little graph/code overhead; "
            "only full_int8 (tflite), int16 and int8 (qualia) shrink activations.",
        }

    def deploy_and_check(self, backend: str, mode: str) -> dict[str, Any]:
        if backend not in BACKENDS:
            return {"error": f"unknown backend {backend!r}, choose from {sorted(BACKENDS)}"}
        if mode not in BACKENDS[backend]["modes"]:
            return {"error": f"mode {mode!r} not available for {backend}: {list(BACKENDS[backend]['modes'])}"}
        res = deploy(self.model_name, backend, mode, self.out_dir / backend)
        check = check_constraints(
            res["flash_bytes"], res["ram_bytes"], res["accuracy"], self.target,
            self.reference[backend], self.max_accuracy_drop,
        )  # fmt: skip
        return {
            "backend": backend,
            "mode": mode,
            "flash_kb": round(res["flash_bytes"] / 1024, 1),
            "ram_kb": round(res["ram_bytes"] / 1024, 1),
            "ram_measured": res["ram_measured"],
            "accuracy": round(res["accuracy"], 4),
            **check,
        }


def options_description() -> str:
    lines = []
    for backend, cfg in BACKENDS.items():
        lines.append(f"- backend `{backend}`:")
        lines.extend(f"    - `{mode}`: {m['description']}" for mode, m in cfg["modes"].items())
    return "\n".join(lines)
