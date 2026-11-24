"""MCP server tests: an in-process MCP client against the server, with a fake scenario."""

import asyncio
import json
from typing import Any, ClassVar

from mcp import Client

from edge_deploy_agent import mcp_server


class FakeScenario:
    backends: ClassVar[list[str]] = ["qualia"]
    chip = None

    def __init__(self, model: str, target: str, max_accuracy_drop: float) -> None:
        self.deploy_calls: list[tuple[str, str]] = []

    def brief(self) -> str:
        return "Deploy toy on tiny."

    def inspect(self) -> dict:
        return {"params": 10}

    def deploy_and_check(self, backend: str, mode: str) -> dict:
        self.deploy_calls.append((backend, mode))
        return {"backend": backend, "mode": mode, "fits": mode == "int8", "violations": []}


def _data(result: Any) -> dict:
    assert not result.is_error, result
    return result.structured_content or json.loads(result.content[0].text)


async def _session() -> tuple[list[str], dict, dict, dict, list, bool]:
    async with Client(mcp_server.server) as client:
        names = [t.name for t in (await client.list_tools()).tools]
        sid = _data(await client.call_tool("start_scenario", {"model": "toy", "target": "tiny"}))["scenario_id"]
        first = _data(await client.call_tool("deploy_and_check", {"scenario_id": sid, "backend": "qualia", "mode": "int8"}))
        again = _data(await client.call_tool("deploy_and_check", {"scenario_id": sid, "backend": "qualia", "mode": "int8"}))
        search = _data(await client.call_tool("search_datasheet", {"scenario_id": sid, "query": "flash"}))
        attempts = _data(await client.call_tool("list_attempts", {"scenario_id": sid}))["attempts"]
        unknown = await client.call_tool("inspect_model", {"scenario_id": "nope"})
        return names, first, again, search, attempts, unknown.is_error


def test_mcp_tools_share_the_agent_rules(monkeypatch) -> None:
    monkeypatch.setattr(mcp_server, "scenario_factory", FakeScenario)
    names, first, again, search, attempts, unknown_is_error = asyncio.run(_session())

    assert {"start_scenario", "inspect_model", "search_datasheet", "set_memory_budget", "deploy_and_check"} <= set(names)
    assert first["fits"] and "note" not in first
    assert again["note"] == "already measured, not redeployed"  # served from memory
    assert len(attempts) == 1
    assert "error" in search  # generic target: no datasheet
    assert unknown_is_error
