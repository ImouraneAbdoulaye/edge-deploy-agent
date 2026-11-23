# Edge Deploy Agent

An LLM agent that figures out **how to deploy a neural network on a microcontroller**: given a trained model and a hardware target, it plans, quantizes, measures and iterates until the model fits the target's flash, RAM and accuracy constraints, or explains why it cannot.

Built with [LangGraph](https://github.com/langchain-ai/langgraph) and open-weight LLMs (local via [Ollama](https://ollama.com), or [Groq](https://groq.com)'s free tier). No paid API needed.

> Status: work in progress. The deployment tools work; the agent loop, datasheet retrieval and evaluation are being built (see roadmap).

## Why

Deploying a model on an MCU is a search problem: which quantization scheme, how much accuracy can be traded, does the activation memory fit in SRAM, and what does the chip actually offer? Engineers solve it by trial and error. This project tests whether an LLM agent with the right tools can do the same search reliably, and at what cost compared with a brute-force baseline.

## Two deployment backends

| Backend | Input | Output | Modes | Footprint |
|---|---|---|---|---|
| **TFLite** | Keras | `.tflite` flatbuffer, run by the TFLite Micro interpreter | `float32`, `float16`, `dynamic_int8`, `full_int8` | flash = file size, RAM estimated from activation buffers |
| **[Qualia-CodeGen](https://github.com/LEAT-EDGE/qualia-codegen-core)** | PyTorch | Plain C library, no runtime | `float32`, `int16`, `int8` (power-of-two fixed point) | flash and RAM measured on the compiled code |

For Qualia, the per-layer fixed-point scale factors are computed from calibration data, with headroom so that accumulators (32-bit for int16, 16-bit for int8) cannot overflow. The generated C code is compiled and run on the host (Qualia's Linux demo) to measure its real accuracy.

## Tools available to the agent

| Tool | What it does |
|---|---|
| `inspect_model` | Parameters, layer shapes, float32 weight size, peak activation memory estimate |
| `deploy` | Converts a model with a backend and mode, returns flash, RAM and measured accuracy |
| `check_constraints` | Compares flash, RAM and accuracy drop with the target's budget |

## Baseline results

Brute force over modes (least aggressive first), accuracy drop ≤ 1 point vs. float32:

| Model | Target | TFLite | Qualia |
|---|---|---|---|
| mnist_cnn_small (5 k params) | tiny (64 KB / 16 KB) | full_int8 | int16 |
| mnist_cnn_small | small (256 KB / 64 KB) | float32 | float32 |
| mnist_cnn_medium (225 k params) | tiny | ✗ | ✗ |
| mnist_cnn_medium | small | full_int8 | int8 |
| mnist_cnn_medium | medium (1 MB / 256 KB) | float32 | float32 |
| fashion_mlp_large (670 k params) | tiny / small | ✗ | ✗ |
| fashion_mlp_large | medium | dynamic_int8 | int8 |

The TFLite RAM estimate for float32 models matches the RAM measured on Qualia's generated buffers (26.4 KB for mnist_cnn_small).

## Quick start

```bash
conda create -n edgeagent python=3.11 -y && conda activate edgeagent
conda install -c conda-forge gcc gxx cmake ninja -y   # to build Qualia's C code
pip install -e ".[dev]"

pytest                                   # tool tests (no download)
python scripts/make_reference_models.py  # trains 3 reference models (Keras + PyTorch)
python scripts/baseline.py               # brute-force baseline, both backends
```

## Roadmap

- [x] Deployment tools: inspection, quantization, evaluation, constraint check
- [x] Qualia-CodeGen backend: PyTorch → C (float32 / int16 / int8), host-measured accuracy and memory
- [x] Reference models and brute-force baseline
- [ ] LangGraph agent: plan → act → observe loop with tool calling and memory of past attempts
- [ ] Retrieval over MCU datasheets (RAG) to derive memory budgets from real chips
- [ ] Evaluation harness: success rate, iterations, tokens and time vs. baseline, across LLMs
