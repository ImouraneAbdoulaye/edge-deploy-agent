"""Talk to the MCP server the way an MCP client does: launch it over stdio, list its tools,
then run a short fixed sequence of calls (no LLM involved).

    python scripts/mcp_demo.py                        # generic target
    python scripts/mcp_demo.py --chip LPC55S69JBD100  # also searches the chip's datasheet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from mcp import Client, StdioServerParameters


def show(title: str, result) -> dict:
    data = result.structured_content or json.loads(result.content[0].text)
    print(f"\n--- {title}\n{json.dumps(data, indent=2)[:1500]}")
    return data


async def main(model: str, target: str, chip: str | None) -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "edge_deploy_agent.mcp_server"])
    async with Client(params) as client:
        tools = (await client.list_tools()).tools
        print("Tools:", ", ".join(t.name for t in tools))

        sid = show("start_scenario", await client.call_tool("start_scenario", {"model": model, "target": target}))[
            "scenario_id"
        ]
        show("inspect_model", await client.call_tool("inspect_model", {"scenario_id": sid}))
        for mode in ("int16", "int16"):  # the second call is served from memory
            show(f"deploy_and_check qualia {mode}", await client.call_tool(
                "deploy_and_check", {"scenario_id": sid, "backend": "qualia", "mode": mode}))  # fmt: skip

        if chip:
            cid = show("start_scenario", await client.call_tool("start_scenario", {"model": model, "target": chip}))[
                "scenario_id"
            ]
            show("search_datasheet", await client.call_tool(
                "search_datasheet", {"scenario_id": cid, "query": f"{chip} ordering information flash SRAM"}))  # fmt: skip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mnist_cnn_small")
    parser.add_argument("--target", default="tiny")
    parser.add_argument("--chip", default=None)
    args = parser.parse_args()
    asyncio.run(main(args.model, args.target, args.chip))
