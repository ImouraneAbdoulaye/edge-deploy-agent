"""Non-agent baseline: try every quantization mode in a fixed order for each scenario.

    python scripts/baseline.py

This is the reference the agent will be compared against: it always finds the
answer when one exists among the available modes, but pays for it by running
every conversion and evaluation. It also checks that the tools work end to end.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from edge_deploy_agent.data import load_dataset
from edge_deploy_agent.tools.convert import QUANT_MODES, convert_model
from edge_deploy_agent.tools.evaluate import check_constraints, evaluate_tflite

ROOT = Path(__file__).resolve().parents[1]
DATASET_OF = {"mnist_cnn_small": "mnist", "mnist_cnn_medium": "mnist", "fashion_mlp_large": "fashion_mnist"}


def run_scenario(model_name: str, target_name: str, target: dict) -> dict:
    (x_train, _), (x_test, y_test) = load_dataset(DATASET_OF[model_name])
    model_path = ROOT / "models" / f"{model_name}.keras"
    out_dir = ROOT / "artifacts" / "baseline"

    attempts = []
    start = time.perf_counter()
    solution = None
    reference_accuracy = None
    for mode in QUANT_MODES:  # least aggressive first; float32 gives the reference accuracy
        conv = convert_model(model_path, mode, out_dir, x_calib=x_train)
        ev = evaluate_tflite(conv["tflite_path"], x_test, y_test)
        if reference_accuracy is None:
            reference_accuracy = ev["accuracy"]
        check = check_constraints(
            conv["flash_bytes"], conv["est_ram_bytes"], ev["accuracy"], target, reference_accuracy
        )
        attempts.append({**conv, "accuracy": ev["accuracy"], **check})
        if check["fits"]:
            solution = mode
            break

    return {
        "model": model_name,
        "target": target_name,
        "solution": solution,
        "num_attempts": len(attempts),
        "seconds": round(time.perf_counter() - start, 1),
        "attempts": attempts,
    }


def main() -> None:
    targets = yaml.safe_load((ROOT / "configs" / "targets.yaml").read_text())
    results = [run_scenario(m, t, targets[t]) for m in DATASET_OF for t in targets]

    print(f"\n{'model':<20} {'target':<8} {'solution':<14} attempts")
    for r in results:
        print(f"{r['model']:<20} {r['target']:<8} {r['solution']!s:<14} {r['num_attempts']}")

    out = ROOT / "artifacts" / "baseline_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nDetails: {out}")


if __name__ == "__main__":
    main()
