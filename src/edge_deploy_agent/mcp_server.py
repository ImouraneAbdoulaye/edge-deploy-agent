"""MCP server: the agent's tools, usable by any MCP client (Claude Desktop, VS Code, other agents).

    python -m edge_deploy_agent.mcp_server            # stdio transport

The client's LLM plays the agent: it opens a scenario (model + target), then calls the same
tools as the LangGraph agent, with the same checks. A memory declaration must quote the
retrieved datasheet passages, and a configuration already measured in a scenario is returned
from memory instead of being redeployed. Scenarios live in the server process.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from edge_deploy_agent.agent.graph import BudgetArgs
from edge_deploy_agent.agent.scenario import (
    BUDGET_POLICY,
    DEFAULT_MAX_ACCURACY_DROP,
    Scenario,
    load_targets,
    options_description,
)
from edge_deploy_agent.tools.deploy import _REGISTRY
from edge_deploy_agent.tools.deploy import register_model as _register_model

server = MCPServer(
    "edge-deploy-agent",
    instructions=(
        "Deploy a neural network on a microcontroller within flash, RAM and accuracy constraints. "
        "Call start_scenario first, then inspect_model, estimate what can fit, and deploy only "
        "promising configurations. For a real chip, find its memory with search_datasheet and "
        "declare it with set_memory_budget before deploying. " + BUDGET_POLICY
    ),
)

scenario_factory = Scenario  # replaced in tests
_scenarios: dict[str, Any] = {}
_attempts: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}

ScenarioId = Annotated[str, Field(description="Id returned by start_scenario")]


def _get(scenario_id: str) -> Any:
    if scenario_id not in _scenarios:
        raise KeyError(f"unknown scenario {scenario_id!r}: call start_scenario first")
    return _scenarios[scenario_id]


@server.tool()
def list_models() -> dict[str, Any]:
    """Models available to deploy, with the backends each one supports."""
    return {"models": {name: spec.backends for name, spec in _REGISTRY.items()}}


@server.tool()
def list_targets() -> dict[str, Any]:
    """Deployment targets: generic memory budgets, and real chips whose memory must be read
    in their datasheet."""
    from edge_deploy_agent.rag.index import load_chips

    return {
        "generic": {name: {k: t[k] for k in ("description", "flash_budget_kb", "ram_budget_kb")}
                    for name, t in load_targets().items()},
        "chips": {name: chip["vendor"] for name, chip in load_chips().items()},
    }  # fmt: skip


@server.tool()
def register_model(
    name: Annotated[str, Field(description="Name to give the model")],
    data_path: Annotated[str, Field(description=".npz with x_test, y_test and optionally x_calib, channels last")],
    keras_path: Annotated[str | None, Field(description=".keras model, enables the TFLite backend")] = None,
    torch_path: Annotated[str | None, Field(description="Full PyTorch model (torch.save), enables Qualia")] = None,
) -> dict[str, Any]:
    """Bring your own model: make a Keras and/or PyTorch model available to start_scenario."""
    spec = _register_model(name, data_path, keras_path=keras_path, torch_path=torch_path)
    return {"model": name, "backends": spec.backends}


@server.tool()
def start_scenario(
    model: Annotated[str, Field(description="Model name, see list_models")],
    target: Annotated[str, Field(description="Generic target or chip part number, see list_targets")],
    max_accuracy_drop: Annotated[
        float, Field(description="Max absolute accuracy drop vs. the float32 model")
    ] = DEFAULT_MAX_ACCURACY_DROP,
) -> dict[str, Any]:
    """Open a deployment scenario and get its brief: constraints, reference accuracy and the
    available backends and quantization modes."""
    scenario = scenario_factory(model, target, max_accuracy_drop)
    scenario_id = uuid.uuid4().hex[:8]
    _scenarios[scenario_id], _attempts[scenario_id] = scenario, {}
    return {
        "scenario_id": scenario_id,
        "brief": scenario.brief(),
        "options": options_description(scenario.backends),
        "needs_datasheet": bool(getattr(scenario, "chip", None)),
    }


@server.tool()
def inspect_model(scenario_id: ScenarioId) -> dict[str, Any]:
    """Parameters, layers, and weight/activation sizes of the model at 32, 16 and 8 bits."""
    return _get(scenario_id).inspect()


@server.tool()
def search_datasheet(
    scenario_id: ScenarioId,
    query: Annotated[str, Field(description="What to look for, e.g. 'ordering information flash memory size'")],
) -> dict[str, Any]:
    """Search the target chip's datasheets. Returns the 3 most relevant passages with their page."""
    scenario = _get(scenario_id)
    if not getattr(scenario, "chip", None):
        return {"error": "this target is generic: its budgets are in the brief, no datasheet to search"}
    return {"passages": scenario.search_datasheet(query)}


@server.tool()
def set_memory_budget(
    scenario_id: ScenarioId,
    part_number_decoding: Annotated[str, Field(description=BudgetArgs.model_fields["part_number_decoding"].description)],
    flash_kb: Annotated[float, Field(description=BudgetArgs.model_fields["flash_kb"].description)],
    sram_kb: Annotated[float, Field(description=BudgetArgs.model_fields["sram_kb"].description)],
    internal_flash: Annotated[bool, Field(description=BudgetArgs.model_fields["internal_flash"].description)],
    evidence_quote: Annotated[str, Field(description=BudgetArgs.model_fields["evidence_quote"].description)],
    evidence: Annotated[str, Field(description=BudgetArgs.model_fields["evidence"].description)],
) -> dict[str, Any]:
    """Declare the chip memory found in the datasheet; returns the model's flash and RAM
    budgets. Rejected unless the quote comes from the search results and states the sizes."""
    scenario = _get(scenario_id)
    if not getattr(scenario, "chip", None):
        return {"error": "this target is generic: its budgets are already set"}
    return scenario.set_memory_budget(
        flash_kb=flash_kb, sram_kb=sram_kb, internal_flash=internal_flash, evidence=evidence,
        part_number_decoding=part_number_decoding, evidence_quote=evidence_quote,
    )  # fmt: skip


@server.tool()
def deploy_and_check(
    scenario_id: ScenarioId,
    backend: Annotated[str, Field(description="Deployment backend: 'tflite' or 'qualia'")],
    mode: Annotated[str, Field(description="Quantization mode, one of the backend's modes")],
) -> dict[str, Any]:
    """Convert the model with a backend and mode, then measure flash, RAM and accuracy and check
    them against the constraints. Expensive: a configuration already measured is not redeployed."""
    scenario, tried = _get(scenario_id), _attempts[scenario_id]
    if (backend, mode) in tried:
        return {**tried[(backend, mode)], "note": "already measured, not redeployed"}
    result = scenario.deploy_and_check(backend, mode)
    if "error" not in result:
        tried[(backend, mode)] = result
    return result


@server.tool()
def list_attempts(scenario_id: ScenarioId) -> dict[str, Any]:
    """Deployments measured so far in this scenario."""
    _get(scenario_id)
    return {"attempts": list(_attempts[scenario_id].values())}


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
