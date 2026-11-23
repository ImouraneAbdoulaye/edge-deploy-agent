"""A deployment scenario (model + target) and the tools the agent can call on it."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from edge_deploy_agent.tools.deploy import BACKENDS, ROOT, deploy
from edge_deploy_agent.tools.evaluate import check_constraints
from edge_deploy_agent.tools.inspect_model import inspect_model

REFERENCE_CACHE = ROOT / "artifacts" / "reference_accuracy.json"
_INDEX = None  # DatasheetIndex, loaded lazily
SEARCH_K = 3  # passages per search: enough for one table, small enough for free-tier token caps
DEFAULT_MAX_ACCURACY_DROP = 0.01


def _normalize(text: str) -> str:
    """Lowercase alphanumerics only, so quotes survive table separators and spacing."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


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


# Share of the chip's memory left to the model; the rest is for firmware, stack, buffers.
MODEL_FLASH_SHARE = 0.5
MODEL_RAM_SHARE = 0.5
# Chips without on-chip flash run from external flash (e.g. QSPI/FlexSPI NOR on the board).
EXTERNAL_FLASH_BUDGET_KB = 4096

BUDGET_POLICY = (
    f"Memory budget policy: the model may use {MODEL_FLASH_SHARE:.0%} of the usable on-chip flash and "
    f"{MODEL_RAM_SHARE:.0%} of the total on-chip SRAM. If the chip has no on-chip flash, the model is "
    f"stored in external flash with a budget of {EXTERNAL_FLASH_BUDGET_KB} KB."
)


@dataclass
class Scenario:
    """A model to deploy on a target.

    ``target_name`` is either a generic profile from configs/targets.yaml (budgets
    given to the agent), or a real chip from configs/chips.yaml: the agent then has
    to find the chip's memory in its datasheets and declare it with
    ``set_memory_budget`` before deploying.
    """

    model_name: str
    target_name: str
    max_accuracy_drop: float = DEFAULT_MAX_ACCURACY_DROP
    out_dir: Path = ROOT / "artifacts" / "agent"
    target: dict[str, Any] | None = field(init=False)
    chip: dict[str, Any] | None = field(init=False)
    extracted: dict[str, Any] | None = field(init=False, default=None)
    retrieved: list[str] = field(init=False, default_factory=list)
    rejected_declarations: int = field(init=False, default=0)
    reference: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        from edge_deploy_agent.rag.index import load_chips

        chips = load_chips()
        self.chip = chips.get(self.target_name)
        self.target = None if self.chip else load_targets()[self.target_name]
        self.reference = {b: reference_accuracy(self.model_name, b) for b in BACKENDS}

    def brief(self) -> str:
        """Task description given to the agent as the first user message."""
        refs = ", ".join(f"{b}: {acc:.3f}" for b, acc in self.reference.items())
        if self.chip:
            target_lines = (
                f"Deploy the model `{self.model_name}` on the microcontroller `{self.target_name}` "
                f"({self.chip['vendor']}).\n"
                "- Its memory sizes are not given: find this exact part's usable on-chip flash and total "
                "SRAM in its datasheet with `search_datasheet`, then declare them with `set_memory_budget`.\n"
                f"- {BUDGET_POLICY}\n"
            )
        else:
            target_lines = (
                f"Deploy the model `{self.model_name}` on the target `{self.target_name}` "
                f"({self.target['description']}).\n"
                f"- Flash budget for the model: {self.target['flash_budget_kb']} KB\n"
                f"- RAM budget for the model: {self.target['ram_budget_kb']} KB\n"
            )
        return (
            target_lines
            + f"- Max accuracy drop vs. the float32 model: {self.max_accuracy_drop:.3f} (absolute)\n"
            + f"- Float32 reference accuracy per backend: {refs}\n"
            + "Find a backend and mode that satisfies all three constraints, or show that none exists."
        )

    # ---- tools -----------------------------------------------------------------

    def search_datasheet(self, query: str) -> list[dict[str, Any]]:
        from edge_deploy_agent.rag.index import DatasheetIndex

        global _INDEX
        if _INDEX is None:
            _INDEX = DatasheetIndex()
        hits = _INDEX.search(query, chip=self.target_name, k=SEARCH_K)
        self.retrieved.extend(h.text for h in hits)
        return [{"source": f"{h.doc} p.{h.page}", "text": h.text} for h in hits]

    def check_grounding(self, quote: str, flash_kb: float, sram_kb: float, internal_flash: bool) -> list[str]:
        """Problems with a memory declaration: the quote must come from retrieved passages and
        contain the declared sizes (in KB, or in MB for whole megabytes)."""
        problems = []
        sources = [_normalize(t) for t in self.retrieved]
        for segment in (s for s in quote.split("...") if s.strip()):
            if not any(_normalize(segment) in src for src in sources):
                problems.append(f"quote not found verbatim in the search results: {segment.strip()[:120]!r}")
        numbers = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", quote)}
        sizes = numbers | {n * 1024 for n in numbers}
        if internal_flash and flash_kb not in sizes:
            problems.append(f"flash {flash_kb:g} KB does not appear in the quote")
        if sram_kb not in sizes:
            problems.append(f"SRAM {sram_kb:g} KB does not appear in the quote")
        return problems

    def set_memory_budget(
        self, flash_kb: float, sram_kb: float, internal_flash: bool, evidence: str,
        part_number_decoding: str = "", evidence_quote: str | None = None,
    ) -> dict:
        """Record the memory sizes the agent read in the datasheet and derive the model budget.

        With ``evidence_quote``, the declaration is rejected (budget left unset) unless it is
        grounded in the retrieved passages.
        """
        if evidence_quote is not None:
            problems = self.check_grounding(evidence_quote, flash_kb, sram_kb, internal_flash)
            if problems:
                self.rejected_declarations += 1
                return {"error": "declaration rejected, budget not set", "problems": problems}
        flash_budget = flash_kb * MODEL_FLASH_SHARE if internal_flash else EXTERNAL_FLASH_BUDGET_KB
        self.extracted = {
            "part_number_decoding": part_number_decoding, "flash_kb": flash_kb, "sram_kb": sram_kb,
            "internal_flash": internal_flash, "evidence": evidence,
        }  # fmt: skip
        self.target = {
            "description": f"{self.target_name}, from datasheet",
            "flash_budget_kb": round(flash_budget, 1),
            "ram_budget_kb": round(sram_kb * MODEL_RAM_SHARE, 1),
        }
        return {"model_flash_budget_kb": self.target["flash_budget_kb"], "model_ram_budget_kb": self.target["ram_budget_kb"]}

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
        if self.target is None:
            return {"error": "memory budget unknown: find it in the datasheet and call set_memory_budget first"}
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
