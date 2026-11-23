"""Non-agent baseline: for each scenario and backend, try every mode in a fixed order.

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

from edge_deploy_agent.tools.deploy import BACKENDS, MODEL_DATASET, deploy
from edge_deploy_agent.tools.evaluate import check_constraints

ROOT = Path(__file__).resolve().parents[1]


def run_scenario(model_name: str, target_name: str, target: dict, backend: str) -> dict:
    out_dir = ROOT / "artifacts" / "baseline" / backend
    attempts = []
    start = time.perf_counter()
    solution = None
    reference_accuracy = None
    for mode in BACKENDS[backend]["modes"]:  # float32 first: it gives the reference accuracy
        res = deploy(model_name, backend, mode, out_dir)
        if reference_accuracy is None:
            reference_accuracy = res["accuracy"]
        check = check_constraints(res["flash_bytes"], res["ram_bytes"], res["accuracy"], target, reference_accuracy)
        attempts.append({**res, **check})
        if check["fits"]:
            solution = mode
            break

    return {
        "model": model_name,
        "target": target_name,
        "backend": backend,
        "solution": solution,
        "num_attempts": len(attempts),
        "seconds": round(time.perf_counter() - start, 1),
        "attempts": attempts,
    }


def main() -> None:
    targets = yaml.safe_load((ROOT / "configs" / "targets.yaml").read_text())
    results = [
        run_scenario(m, t, targets[t], b) for m in MODEL_DATASET for t in targets for b in BACKENDS
    ]

    print(f"\n{'model':<20} {'target':<8} {'tflite':<14} {'qualia':<14}")
    for i in range(0, len(results), len(BACKENDS)):
        by_backend = {r["backend"]: r for r in results[i : i + len(BACKENDS)]}
        r = by_backend["tflite"]
        print(
            f"{r['model']:<20} {r['target']:<8} {by_backend['tflite']['solution']!s:<14} "
            f"{by_backend['qualia']['solution']!s:<14}"
        )

    out = ROOT / "artifacts" / "baseline_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nDetails: {out}")


if __name__ == "__main__":
    main()
