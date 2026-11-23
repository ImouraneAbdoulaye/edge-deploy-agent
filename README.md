# Edge Deploy Agent

An LLM agent that figures out **how to deploy a neural network on a microcontroller**: given a trained model and a hardware target, it plans, quantizes, measures and iterates until the model fits the target's flash, RAM and accuracy constraints, or explains why it cannot.

Built with [LangGraph](https://github.com/langchain-ai/langgraph) and open-weight LLMs (local via [Ollama](https://ollama.com), or [Groq](https://groq.com)'s free tier). No paid API needed.

> Status: work in progress. The deployment tools work; the agent loop, datasheet retrieval and evaluation are being built (see roadmap).

## Why

Deploying a model on an MCU is a search problem: which quantization scheme, how much accuracy can be traded, does the activation memory fit in SRAM, and what does the chip actually offer? Engineers solve it by trial and error. This project tests whether an LLM agent with the right tools can do the same search reliably, and at what cost compared with a brute-force baseline.

## Tools available to the agent

| Tool | What it does |
|---|---|
| `inspect_model` | Parameters, layer shapes, float32 weight size, peak activation memory estimate |
| `convert_model` | Keras → TFLite in `float32`, `float16`, `dynamic_int8` or `full_int8` (with calibration) |
| `evaluate_tflite` | Top-1 accuracy of the converted model with the TFLite interpreter |
| `check_constraints` | Compares flash, RAM and accuracy with the target's budget |

## Quick start

```bash
conda create -n edgeagent python=3.11 -y && conda activate edgeagent
pip install -e ".[dev]"

pytest                                   # tool tests (no download)
python scripts/make_reference_models.py  # trains 3 reference models
python scripts/baseline.py               # brute-force baseline over all scenarios
```

## Roadmap

- [x] Deployment tools: inspection, quantization, evaluation, constraint check
- [x] Reference models and brute-force baseline
- [ ] LangGraph agent: plan → act → observe loop with tool calling and memory of past attempts
- [ ] Retrieval over MCU datasheets (RAG) to derive memory budgets from real chips
- [ ] Evaluation harness: success rate, iterations, tokens and time vs. baseline, across LLMs
