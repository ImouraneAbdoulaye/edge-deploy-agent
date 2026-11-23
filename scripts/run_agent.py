"""Run the agent on one scenario and print its reasoning.

    python scripts/run_agent.py mnist_cnn_small tiny
    python scripts/run_agent.py fashion_mlp_large medium --provider ollama --llm qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json

from langchain_core.messages import AIMessage, ToolMessage

from edge_deploy_agent.agent.graph import run_agent
from edge_deploy_agent.agent.llm import get_llm
from edge_deploy_agent.agent.scenario import Scenario
from edge_deploy_agent.tools.deploy import ROOT


def print_transcript(messages: list) -> None:
    for m in messages:
        if isinstance(m, AIMessage):
            if m.content:
                print(f"\n[agent] {m.content.strip()}")
            for call in m.tool_calls:
                print(f"[agent -> {call['name']}] {json.dumps(call['args'])}")
        elif isinstance(m, ToolMessage):
            print(f"[{m.name}] {m.content[:400]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("target")
    parser.add_argument("--provider", default="groq", choices=["groq", "ollama"])
    parser.add_argument("--llm", default=None, help="LLM name (defaults per provider)")
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    scenario = Scenario(args.model, args.target)
    print(scenario.brief())
    result = run_agent(get_llm(args.provider, args.llm), scenario, args.max_steps)
    print_transcript(result["transcript"])

    print("\n=== Result ===")
    print(json.dumps(result["answer"], indent=2))
    print(
        f"deployments: {result['num_deployments']} | LLM calls: {result['llm_calls']} | "
        f"tokens in/out: {result['input_tokens']}/{result['output_tokens']} | {result['seconds']} s"
    )
    out = ROOT / "artifacts" / "runs" / f"{args.model}__{args.target}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: v for k, v in result.items() if k != "transcript"}, indent=2))


if __name__ == "__main__":
    main()
