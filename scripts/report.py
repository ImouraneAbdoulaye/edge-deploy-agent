"""Summarize artifacts/eval/results.jsonl into a Markdown table.

    python scripts/report.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean

from edge_deploy_agent.tools.deploy import ROOT

RESULTS = ROOT / "artifacts" / "eval" / "results.jsonl"
REPORT = ROOT / "artifacts" / "eval" / "report.md"


def pct(values: list[bool]) -> str:
    return f"{100 * sum(values) / len(values):.0f} %" if values else "-"


def main() -> None:
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines()]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["llm"], r["kind"])].append(r)

    lines = [
        (
            "| LLM | Scenarios | Runs | Correct | Memory extracted | Deployments (agent / brute force) "
            "| Tokens / run | Time / run | Errors |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    # Correct = share of all runs (errors count as failures); other columns average successful runs.
    for (llm, kind), runs in sorted(groups.items()):
        ok = [r for r in runs if r["status"] == "ok"]
        errors = len(runs) - len(ok)
        if not ok:
            lines.append(f"| {llm} | {kind} | {len(runs)} | - | - | - | - | - | {errors} |")
            continue
        extraction = [r["extraction_correct"] for r in ok if r["extraction_correct"] is not None]
        lines.append(
            f"| {llm} | {kind} | {len(runs)} | {pct([r.get('correct', False) for r in runs])} "
            f"| {pct(extraction) if extraction else 'n/a'} "
            f"| {mean(r['num_deployments'] for r in ok):.1f} / {mean(r['baseline_deployments'] for r in ok):.1f} "
            f"| {mean(r['input_tokens'] + r['output_tokens'] for r in ok) / 1000:.1f} k "
            f"| {mean(r['seconds'] for r in ok):.0f} s | {errors} |"
        )

    failures = [r for r in rows if r["status"] == "ok" and not r["correct"]]
    if failures:
        lines += ["", "Wrong answers:", ""]
        for r in failures:
            ans = r["answer"]
            got = "no answer" if ans is None else (f"{ans['backend']}/{ans['mode']}" if ans["feasible"] else "infeasible")
            ext = r.get("extracted_memory")
            mem = f", extracted {ext['flash_kb']:g} KB / {ext['sram_kb']:g} KB" if ext else ""
            lines.append(f"- {r['llm']} · {r['model']} → {r['target']}: {got}{mem}")

    text = "\n".join(lines)
    REPORT.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
