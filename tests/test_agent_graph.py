"""Agent graph tests with a scripted fake LLM and a fake scenario (no network, no deployment)."""

from typing import Any, ClassVar

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from edge_deploy_agent.agent.graph import run_agent


class ScriptedLLM(FakeMessagesListChatModel):
    """Replays scripted AIMessages; tool binding is a no-op."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedLLM":
        return self


class FakeScenario:
    model_name, target_name = "toy", "tiny"

    def __init__(self) -> None:
        self.deploy_calls: list[tuple[str, str]] = []

    def brief(self) -> str:
        return "Deploy toy on tiny."

    def inspect(self) -> dict:
        return {"params": 10}

    def deploy_and_check(self, backend: str, mode: str) -> dict:
        self.deploy_calls.append((backend, mode))
        fits = mode == "int8"
        return {
            "backend": backend, "mode": mode, "flash_kb": 1.0, "ram_kb": 1.0, "ram_measured": True,
            "accuracy": 0.9, "fits": fits, "violations": [] if fits else ["RAM too big"],
        }  # fmt: skip


def call(name: str, args: dict, i: int) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


def test_agent_loop_records_attempts_dedupes_and_stops_on_answer() -> None:
    llm = ScriptedLLM(
        responses=[
            call("inspect_model", {}, 0),
            call("deploy_and_check", {"backend": "qualia", "mode": "int16"}, 1),
            call("deploy_and_check", {"backend": "qualia", "mode": "int16"}, 2),  # duplicate
            call("deploy_and_check", {"backend": "qualia", "mode": "int8"}, 3),
            call("submit_answer", {"feasible": True, "backend": "qualia", "mode": "int8", "rationale": "fits"}, 4),
        ]
    )
    scenario = FakeScenario()
    result = run_agent(llm, scenario)

    assert result["answer"] == {"feasible": True, "backend": "qualia", "mode": "int8", "rationale": "fits"}
    assert scenario.deploy_calls == [("qualia", "int16"), ("qualia", "int8")]  # duplicate not redeployed
    assert result["num_deployments"] == 2
    assert result["llm_calls"] == 5


def test_agent_stops_at_step_budget() -> None:
    llm = ScriptedLLM(responses=[call("inspect_model", {}, i) for i in range(10)])
    result = run_agent(llm, FakeScenario(), max_steps=3)
    assert result["answer"] is None
    assert result["llm_calls"] == 3


class FakeChipScenario(FakeScenario):
    chip: ClassVar[dict] = {"vendor": "ACME"}

    def search_datasheet(self, query: str) -> list[dict]:
        return [{"source": "ds.pdf p.1", "text": "64 KB flash, 16 KB SRAM " * 50}]

    def set_memory_budget(self, **kwargs: object) -> dict:
        return {"model_flash_budget_kb": 32.0, "model_ram_budget_kb": 8.0}


def test_search_results_pruned_once_budget_accepted() -> None:
    budget = {"part_number_decoding": "X", "flash_kb": 64, "sram_kb": 16, "internal_flash": True,
              "evidence_quote": "64 KB flash, 16 KB SRAM", "evidence": "p.1"}  # fmt: skip
    llm = ScriptedLLM(
        responses=[
            call("search_datasheet", {"query": "memory"}, 0),
            call("set_memory_budget", budget, 1),
            call("submit_answer", {"feasible": False, "backend": "none", "mode": "none", "rationale": "x"}, 2),
        ]
    )
    result = run_agent(llm, FakeChipScenario())
    searches = [m for m in result["transcript"] if getattr(m, "name", None) == "search_datasheet"]
    assert len(searches) == 1
    assert "removed from context" in searches[0].content
    assert result["answer"]["backend"] is None
