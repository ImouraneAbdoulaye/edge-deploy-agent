"""LangGraph agent: reason -> call a tool -> observe, until it submits an answer.

    START -> agent --(tool calls)--> tools --(answer submitted or budget spent)--> END
               ^                       |
               +-----------------------+

Memory: every deployment result is stored in the graph state (``attempts``) and
re-injected into the prompt as a compact table at each step, so the model reasons
over what it has already measured instead of re-reading the whole transcript.
Asking for a configuration that was already measured returns the stored result
without redeploying.
"""

from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from edge_deploy_agent.agent.scenario import Scenario, options_description

SYSTEM_PROMPT = """You are an embedded machine learning engineer. Your job is to deploy a trained \
neural network on a microcontroller within its flash, RAM and accuracy constraints.

Available deployment options:
{options}

How to work:
1. Inspect the model first and estimate, for each option, whether the weights (flash) and \
activations (RAM) can fit the budgets. Write this reasoning down briefly.
2. Each deployment is expensive (conversion, compilation, evaluation on the test set). Only \
deploy options that your estimates say can fit, most promising first. Prefer the least \
aggressive quantization that fits, since it keeps accuracy highest.
3. Use the measured results to update your reasoning. Never deploy the same option twice.
4. Submit your answer with `submit_answer` as soon as one deployment fits, or as soon as your \
estimates and measurements show that no option can fit (for example, if even the smallest \
weights exceed the flash budget). Give a short, quantitative justification.
"""


class DeployArgs(BaseModel):
    backend: Literal["tflite", "qualia"] = Field(description="Deployment backend")
    mode: str = Field(description="Quantization mode, must be one of the backend's modes")


class AnswerArgs(BaseModel):
    feasible: bool = Field(description="True if a measured deployment satisfies all constraints")
    backend: str | None = Field(default=None, description="Chosen backend if feasible")
    mode: str | None = Field(default=None, description="Chosen mode if feasible")
    rationale: str = Field(description="Short quantitative justification")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    attempts: Annotated[list[dict[str, Any]], operator.add]
    answer: dict[str, Any] | None
    steps: int


def _attempts_table(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return "No deployment measured yet."
    rows = ["backend | mode | flash KB | RAM KB | accuracy | fits | violations"]
    for a in attempts:
        rows.append(
            f"{a['backend']} | {a['mode']} | {a['flash_kb']} | {a['ram_kb']} | {a['accuracy']} | "
            f"{a['fits']} | {'; '.join(a['violations']) or '-'}"
        )
    return "\n".join(rows)


def build_agent(llm: BaseChatModel, scenario: Scenario, max_steps: int = 10):
    """Compile the graph for one scenario. ``max_steps`` bounds the number of LLM calls."""
    tools = [
        StructuredTool.from_function(
            func=lambda: None, name="inspect_model",
            description="Parameters, layers, and weight/activation sizes of the model at 32, 16 and 8 bits.",
        ),
        StructuredTool.from_function(
            func=lambda backend, mode: None, name="deploy_and_check", args_schema=DeployArgs,
            description="Convert the model with a backend and mode, then measure flash, RAM and accuracy "
                        "and check them against the constraints. Expensive.",
        ),
        StructuredTool.from_function(
            func=lambda **kw: None, name="submit_answer", args_schema=AnswerArgs,
            description="Submit the final answer and stop.",
        ),
    ]  # fmt: skip
    llm_with_tools = llm.bind_tools(tools)
    system = SYSTEM_PROMPT.format(options=options_description())

    def agent(state: AgentState) -> dict[str, Any]:
        memory = SystemMessage(f"Deployments measured so far:\n{_attempts_table(state['attempts'])}")
        response = llm_with_tools.invoke([SystemMessage(system), *state["messages"], memory])
        return {"messages": [response], "steps": state["steps"] + 1}

    def run_tools(state: AgentState) -> dict[str, Any]:
        last: AIMessage = state["messages"][-1]
        messages, new_attempts, answer = [], [], None
        tried = {(a["backend"], a["mode"]): a for a in state["attempts"]}
        for call in last.tool_calls:
            name, args = call["name"], call["args"]
            if name == "inspect_model":
                result: Any = scenario.inspect()
            elif name == "deploy_and_check":
                key = (args.get("backend"), args.get("mode"))
                if key in tried:
                    result = {**tried[key], "note": "already measured, not redeployed"}
                else:
                    result = scenario.deploy_and_check(*key)
                    if "error" not in result:
                        new_attempts.append(result)
                        tried[key] = result
            elif name == "submit_answer":
                answer = AnswerArgs(**args).model_dump()
                result = "Answer recorded."
            else:
                result = {"error": f"unknown tool {name}"}
            messages.append(ToolMessage(json.dumps(result), tool_call_id=call["id"], name=name))
        return {"messages": messages, "attempts": new_attempts, "answer": answer}

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END  # the model answered in plain text without submitting

    def after_tools(state: AgentState) -> str:
        if state["answer"] is not None or state["steps"] >= max_steps:
            return END
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["agent", END])
    return graph.compile()


def run_agent(llm: BaseChatModel, scenario: Scenario, max_steps: int = 10) -> dict[str, Any]:
    """Run the agent on ``scenario`` and return its answer, attempts, transcript and cost."""
    app = build_agent(llm, scenario, max_steps)
    start = time.perf_counter()
    state = app.invoke(
        {"messages": [HumanMessage(scenario.brief())], "attempts": [], "answer": None, "steps": 0},
        {"recursion_limit": 4 * max_steps + 5},
    )
    usage = [m.usage_metadata for m in state["messages"] if isinstance(m, AIMessage) and m.usage_metadata]
    return {
        "model": scenario.model_name,
        "target": scenario.target_name,
        "answer": state["answer"],
        "attempts": state["attempts"],
        "num_deployments": len(state["attempts"]),
        "llm_calls": state["steps"],
        "input_tokens": sum(u["input_tokens"] for u in usage),
        "output_tokens": sum(u["output_tokens"] for u in usage),
        "seconds": round(time.perf_counter() - start, 1),
        "transcript": state["messages"],
    }
