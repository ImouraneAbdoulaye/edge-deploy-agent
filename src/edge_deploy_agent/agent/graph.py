"""LangGraph agent: reason -> call a tool -> observe, until it submits an answer.

    START -> agent --(tool calls)--> tools --(answer submitted or budget spent)--> END
               ^                       |
               +-----------------------+

Memory: every deployment result is stored in the graph state (``attempts``) and
re-injected into the system prompt as a compact table at each step, so the model reasons
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
from pydantic import BaseModel, Field, ValidationError

from edge_deploy_agent.agent.scenario import Scenario, options_description

SYSTEM_PROMPT = """You are an embedded machine learning engineer. Your job is to deploy a trained \
neural network on a microcontroller within its flash, RAM and accuracy constraints.

Available deployment options:
{options}

How to work:
0. If the target is a microcontroller whose memory sizes are not given, search its datasheet \
first. Datasheets often cover several part numbers: decode the exact part number with the \
datasheet's ordering information scheme (each character encodes an option, often the memory \
size) and make sure the flash and SRAM values apply to that exact part. Use the usable on-chip \
flash (some is sometimes reserved), the total SRAM, and external flash only when there is no \
on-chip flash. Call `set_memory_budget`, citing the source pages, before any deployment.
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


MAX_TOOL_CALL_RETRIES = 2


class DeployArgs(BaseModel):
    backend: Literal["tflite", "qualia"] = Field(description="Deployment backend")
    mode: str = Field(description="Quantization mode, must be one of the backend's modes")


class SearchArgs(BaseModel):
    query: str = Field(description="What to look for, e.g. 'ordering information flash memory size'")


class BudgetArgs(BaseModel):
    # First field on purpose: the model writes the decoding before the numbers it implies.
    part_number_decoding: str = Field(
        description="The exact part number split into its ordering-scheme fields, each with its meaning "
        "from the datasheet, e.g. for a fictional 'XY10R4': 'XY=family, 10=line, R=pin count 64, "
        "4=memory 128 KB flash / 32 KB RAM'. Say so if the datasheet has no ordering scheme."
    )
    flash_kb: float = Field(description="Usable on-chip flash of this exact part in KB (0 if none)")
    sram_kb: float = Field(description="Total on-chip SRAM of this exact part in KB")
    internal_flash: bool = Field(description="False if the chip has no on-chip flash")
    evidence_quote: str = Field(
        description="The datasheet line(s) stating these sizes for this exact part, copied verbatim from "
        "the search results; separate non-adjacent lines with '...'. Checked automatically."
    )
    evidence: str = Field(description="Short explanation with the source pages")


class AnswerArgs(BaseModel):
    feasible: bool = Field(description="True if a measured deployment satisfies all constraints")
    backend: str = Field(description="Chosen backend if feasible, 'none' otherwise")
    mode: str = Field(description="Chosen mode if feasible, 'none' otherwise")
    rationale: str = Field(description="Short quantitative justification")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    attempts: Annotated[list[dict[str, Any]], operator.add]
    answer: dict[str, Any] | None
    steps: int
    tool_call_errors: Annotated[int, operator.add]
    invalid_args: Annotated[int, operator.add]


PRUNED = "[search results removed from context once the memory budget was accepted]"


def _prune_search_results(messages: list[AnyMessage]) -> list[ToolMessage]:
    """Replacements (same message id) that drop earlier datasheet passages from the context.

    Once the memory declaration has passed the grounding check, the passages are no longer
    needed, and keeping them would resend thousands of tokens at every later step.
    """
    return [
        ToolMessage(PRUNED, tool_call_id=m.tool_call_id, name=m.name, id=m.id)
        for m in messages
        if isinstance(m, ToolMessage) and m.name == "search_datasheet" and m.content != PRUNED
    ]


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
    if getattr(scenario, "chip", None):
        tools += [
            StructuredTool.from_function(
                func=lambda query: None, name="search_datasheet", args_schema=SearchArgs,
                description="Search the target chip's datasheets. Returns the 3 most relevant passages "
                            "with their source page.",
            ),
            StructuredTool.from_function(
                func=lambda **kw: None, name="set_memory_budget", args_schema=BudgetArgs,
                description="Declare the chip memory found in the datasheet; returns the model's flash "
                            "and RAM budgets. Required before any deployment.",
            ),
        ]  # fmt: skip
    llm_with_tools = llm.bind_tools(tools)
    system = SYSTEM_PROMPT.format(options=options_description())

    tool_names = {t.name for t in tools}

    def agent(state: AgentState) -> dict[str, Any]:
        # Memory goes into the leading system prompt: a system message placed after tool
        # results breaks some chat formats (gpt-oss then emits malformed tool calls).
        memory = f"\n\nDeployments measured so far:\n{_attempts_table(state['attempts'])}"
        prompt = [SystemMessage(system + memory), *state["messages"]]
        errors = 0
        # Some models occasionally emit a malformed tool call (e.g. gpt-oss calling a tool named
        # "json"); the provider rejects it. Tell the model what went wrong and let it retry.
        for _ in range(MAX_TOOL_CALL_RETRIES + 1):
            try:
                response = llm_with_tools.invoke(prompt)
            except Exception as exc:  # provider-specific error types
                if not any(code in str(exc) for code in ("tool_use_failed", "output_parse_failed")):
                    raise
                errors += 1
                prompt = [*prompt, HumanMessage(
                    f"Your last tool call was invalid. Call exactly one of these tools by name: {sorted(tool_names)}."
                )]  # fmt: skip
                continue
            return {"messages": [response], "steps": state["steps"] + 1, "tool_call_errors": errors}
        raise RuntimeError(f"Model produced {errors} invalid tool calls in a row.")

    def run_tools(state: AgentState) -> dict[str, Any]:
        last: AIMessage = state["messages"][-1]
        messages, new_attempts, answer, invalid_args = [], [], None, 0
        tried = {(a["backend"], a["mode"]): a for a in state["attempts"]}
        for call in last.tool_calls:
            name, args = call["name"], call["args"]
            try:
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
                elif name == "search_datasheet":
                    result = scenario.search_datasheet(args["query"])
                elif name == "set_memory_budget":
                    result = scenario.set_memory_budget(**BudgetArgs(**args).model_dump())
                    if "error" not in result:
                        messages.extend(_prune_search_results(state["messages"]))
                elif name == "submit_answer":
                    submitted = AnswerArgs(**{k: v if v is not None else "none" for k, v in args.items()}).model_dump()
                    for key in ("backend", "mode"):
                        if str(submitted[key]).lower() in ("none", "null", ""):
                            submitted[key] = None
                    fitting = {(a["backend"], a["mode"]) for a in tried.values() if a.get("fits")}
                    if submitted["feasible"] and (submitted["backend"], submitted["mode"]) not in fitting:
                        result = {"error": "rejected: a feasible answer must set the `backend` and `mode` fields to a "
                                  "configuration you measured as fitting, e.g. "
                                  + " or ".join(f'backend="{b}", mode="{m}"' for b, m in sorted(fitting))}
                    else:
                        answer = submitted
                        result = "Answer recorded."
                else:
                    result = {"error": f"unknown tool {name}"}
            except (ValidationError, KeyError, TypeError) as exc:
                # Malformed arguments (common with small models): report them so the model can fix the call.
                invalid_args += 1
                result = {"error": f"invalid arguments for {name}: {exc}"}
            messages.append(ToolMessage(json.dumps(result), tool_call_id=call["id"], name=name))
        return {"messages": messages, "attempts": new_attempts, "answer": answer, "invalid_args": invalid_args}

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        # Plain-text reply without submitting: remind it, within the step budget.
        return "remind" if state["steps"] < max_steps else END

    def remind(state: AgentState) -> dict[str, Any]:
        return {"messages": [HumanMessage("Continue with a tool call; finish with `submit_answer`.")]}

    def after_tools(state: AgentState) -> str:
        if state["answer"] is not None or state["steps"] >= max_steps:
            return END
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("remind", remind)
    graph.add_edge("remind", "agent")
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", "remind", END])
    graph.add_conditional_edges("tools", after_tools, ["agent", END])
    return graph.compile()


def run_agent(llm: BaseChatModel, scenario: Scenario, max_steps: int = 10) -> dict[str, Any]:
    """Run the agent on ``scenario`` and return its answer, attempts, transcript and cost."""
    app = build_agent(llm, scenario, max_steps)
    start = time.perf_counter()
    state = app.invoke(
        {"messages": [HumanMessage(scenario.brief())], "attempts": [], "answer": None, "steps": 0,
         "tool_call_errors": 0, "invalid_args": 0},
        {"recursion_limit": 4 * max_steps + 5},
    )
    usage = [m.usage_metadata for m in state["messages"] if isinstance(m, AIMessage) and m.usage_metadata]
    return {
        "model": scenario.model_name,
        "target": scenario.target_name,
        "answer": state["answer"],
        "extracted_memory": getattr(scenario, "extracted", None),
        "rejected_declarations": getattr(scenario, "rejected_declarations", 0),
        "budget": getattr(scenario, "target", None),
        "attempts": state["attempts"],
        "num_deployments": len(state["attempts"]),
        "llm_calls": state["steps"],
        "tool_call_errors": state["tool_call_errors"],
        "invalid_args": state["invalid_args"],
        "input_tokens": sum(u["input_tokens"] for u in usage),
        "output_tokens": sum(u["output_tokens"] for u in usage),
        "seconds": round(time.perf_counter() - start, 1),
        "transcript": state["messages"],
    }
