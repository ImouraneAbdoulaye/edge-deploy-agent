"""Run the agent on every scenario and score it. Resumable: finished runs are skipped.

    python scripts/evaluate.py --provider ollama --llm qwen2.5:3b --runs 3
    python scripts/evaluate.py --provider groq --llm openai/gpt-oss-120b --runs 1 --kind chip

Results are appended to artifacts/eval/results.jsonl, one line per run. A daily or
per-minute provider quota stops the loop cleanly; run the same command later to resume.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

from edge_deploy_agent.agent.graph import run_agent
from edge_deploy_agent.agent.llm import DEFAULT_MODELS, get_llm
from edge_deploy_agent.agent.scenario import Scenario, cached_deploy
from edge_deploy_agent.evaluation import ALL_CONFIGS, scenarios, score
from edge_deploy_agent.tools.deploy import MODEL_DATASET, ROOT

RESULTS = ROOT / "artifacts" / "eval" / "results.jsonl"
QUOTA_MARKERS = ("rate_limit_exceeded", "tokens per day", "Rate limit reached")


def done_keys() -> set[tuple]:
    if not RESULTS.exists():
        return set()
    keys = set()
    for line in RESULTS.read_text().splitlines():
        r = json.loads(line)
        if r["status"] == "ok":
            keys.add((r["llm"], r["model"], r["target"], r["run"]))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="groq", choices=["groq", "ollama"])
    parser.add_argument("--llm", default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--kind", default="all", choices=["all", "generic", "chip"])
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()
    llm_name = args.llm or DEFAULT_MODELS[args.provider]

    # Measure every configuration once so that agent deployments and scoring are cache hits.
    for model in MODEL_DATASET:
        for backend, mode in ALL_CONFIGS:
            cached_deploy(model, backend, mode)

    llm = get_llm(args.provider, llm_name)
    done = done_keys()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    todo = [(m, t, k, r) for m, t, k in scenarios(args.kind) for r in range(args.runs)
            if (llm_name, m, t, r) not in done]  # fmt: skip
    print(f"{llm_name}: {len(todo)} runs to do")

    for i, (model, target, kind, run_idx) in enumerate(todo, start=1):
        record = {"llm": llm_name, "provider": args.provider, "model": model, "target": target,
                  "kind": kind, "run": run_idx, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}  # fmt: skip
        try:
            result = run_agent(llm, Scenario(model, target), max_steps=args.max_steps)
            result.pop("transcript")
            record.update(status="ok", **result, **score(result))
            verdict = "correct" if record["correct"] else "WRONG"
            print(f"[{i}/{len(todo)}] {model} -> {target}: {verdict}, {record['num_deployments']} deployments "
                  f"(baseline {record['baseline_deployments']}), {record['seconds']} s")  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - recorded, the evaluation goes on
            message = f"{type(exc).__name__}: {exc}"
            if any(marker in message for marker in QUOTA_MARKERS):
                print(f"Provider quota reached, stopping. Resume later with the same command.\n{message[:300]}")
                break
            record.update(status="error", error=message[:2000], traceback=traceback.format_exc()[-2000:])
            print(f"[{i}/{len(todo)}] {model} -> {target}: ERROR {message[:200]}")
        with RESULTS.open("a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
