"""Ground truth and scoring for agent runs.

Every (model, backend, mode) configuration is measured once (cached), so for any
scenario we know exactly which configurations fit. Chip scenarios are judged against
the budget derived from the hand-checked datasheet values in configs/chips.yaml, not
against the budget the agent extracted.
"""

from __future__ import annotations

from typing import Any

from edge_deploy_agent.agent.scenario import (
    budget_from_memory,
    cached_deploy,
    load_targets,
    reference_accuracy,
)
from edge_deploy_agent.rag.index import load_chips
from edge_deploy_agent.tools.deploy import BACKENDS, MODEL_DATASET
from edge_deploy_agent.tools.evaluate import check_constraints

# Fixed brute-force order: backend by backend, least aggressive mode first.
ALL_CONFIGS = [(b, m) for b in BACKENDS for m in BACKENDS[b]["modes"]]
EXTRACTION_TOLERANCE = 0.02


def scenarios(kind: str = "all") -> list[tuple[str, str, str]]:
    """(model, target, kind) for kind in {'generic', 'chip'}."""
    out = []
    if kind in ("all", "generic"):
        out += [(m, t, "generic") for m in MODEL_DATASET for t in load_targets()]
    if kind in ("all", "chip"):
        out += [(m, c, "chip") for m in MODEL_DATASET for c in load_chips()]
    return out


def true_target(target_name: str) -> dict[str, Any]:
    chips = load_chips()
    if target_name in chips:
        gt = chips[target_name]["ground_truth"]
        return budget_from_memory(gt["flash_kb"], gt["sram_kb"], gt["internal_flash"])
    return load_targets()[target_name]


def fitting_configs(model_name: str, target_name: str) -> list[tuple[str, str]]:
    target = true_target(target_name)
    fits = []
    for backend, mode in ALL_CONFIGS:
        res = cached_deploy(model_name, backend, mode)
        check = check_constraints(
            res["flash_bytes"], res["ram_bytes"], res["accuracy"], target, reference_accuracy(model_name, backend)
        )
        if check["fits"]:
            fits.append((backend, mode))
    return fits


def baseline_deployments(model_name: str, target_name: str) -> int:
    """Deployments the brute-force search needs: up to the first fit, or all if none fits."""
    fits = set(fitting_configs(model_name, target_name))
    for i, cfg in enumerate(ALL_CONFIGS, start=1):
        if cfg in fits:
            return i
    return len(ALL_CONFIGS)


def extraction_correct(target_name: str, extracted: dict[str, Any] | None) -> bool | None:
    """None for generic targets; for chips, whether flash and SRAM are within 2 % of the truth."""
    chips = load_chips()
    if target_name not in chips:
        return None
    if not extracted:
        return False
    gt = chips[target_name]["ground_truth"]

    def close(value: float, truth: float) -> bool:
        return abs(value - truth) <= EXTRACTION_TOLERANCE * max(truth, 1)

    if bool(extracted["internal_flash"]) != gt["internal_flash"]:
        return False
    flash_ok = close(extracted["flash_kb"], gt["flash_kb"]) if gt["internal_flash"] else True
    return flash_ok and close(extracted["sram_kb"], gt["sram_kb"])


def score(run: dict[str, Any]) -> dict[str, Any]:
    """Judge one agent run (output of ``run_agent``) against the ground truth."""
    model_name, target_name = run["model"], run["target"]
    fits = fitting_configs(model_name, target_name)
    answer = run["answer"]
    if answer is None:
        correct = False
    elif fits:
        correct = bool(answer["feasible"]) and (answer["backend"], answer["mode"]) in fits
    else:
        correct = not answer["feasible"]
    return {
        "truly_feasible": bool(fits),
        "completed": answer is not None,
        "correct": correct,
        "extraction_correct": extraction_correct(target_name, run.get("extracted_memory")),
        "baseline_deployments": baseline_deployments(model_name, target_name),
    }
