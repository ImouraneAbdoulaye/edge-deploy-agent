# Edge Deploy Agent

An LLM agent that figures out **how to deploy a neural network on a microcontroller**: given a trained model and a hardware target, it plans, quantizes, measures and iterates until the model fits the target's flash, RAM and accuracy constraints, or explains why it cannot.

Built with [LangGraph](https://github.com/langchain-ai/langgraph) and open-weight LLMs (local via [Ollama](https://ollama.com), or [Groq](https://groq.com)'s free tier). No paid API needed.

> Status: work in progress. Tools, agent loop and datasheet retrieval work; the systematic evaluation is being built (see roadmap).

## Why

Deploying a model on an MCU is a search problem: which quantization scheme, how much accuracy can be traded, does the activation memory fit in SRAM, and what does the chip actually offer? Engineers solve it by trial and error. This project tests whether an LLM agent with the right tools can do the same search reliably, and at what cost compared with a brute-force baseline.

## Two deployment backends

| Backend | Input | Output | Modes | Footprint |
|---|---|---|---|---|
| **TFLite** | Keras | `.tflite` flatbuffer, run by the TFLite Micro interpreter | `float32`, `float16`, `dynamic_int8`, `full_int8` | flash = file size, RAM estimated from activation buffers |
| **[Qualia-CodeGen](https://github.com/LEAT-EDGE/qualia-codegen-core)** | PyTorch | Plain C library, no runtime | `float32`, `int16`, `int8` (power-of-two fixed point) | flash and RAM measured on the compiled code |

For Qualia, the per-layer fixed-point scale factors follow qualia-core's post-training rules (minmax range, running max of activations over calibration data, weights and biases sharing one scale factor), plus one addition: **accumulator headroom**. qualia-core relies on quantization-aware training; with post-training quantization only, max-precision weights overflow the accumulators:

| Model | int16 without headroom | int16 with | int8 without | int8 with |
|---|---|---|---|---|
| mnist_cnn_small | overflow (crash) | 97.8 % | 81.5 % | 97.8 % |
| mnist_cnn_medium | overflow (crash) | 98.4 % | 18.5 % | 98.5 % |
| fashion_mlp_large | overflow (crash) | 85.8 % | 22.4 % | 86.9 % |

The generated C code is compiled and run on the host (Qualia's Linux demo) to measure its real accuracy.

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

## The agent

A [LangGraph](https://github.com/langchain-ai/langgraph) loop: the LLM reasons, calls a tool (`inspect_model`, `deploy_and_check`, `submit_answer`), observes the result, and repeats until it submits an answer or runs out of steps. Every measured deployment is kept in the graph state and re-injected into the prompt as a table (memory), and asking twice for the same configuration returns the stored result instead of redeploying.

Example runs with `openai/gpt-oss-120b` on Groq's free tier:

| Scenario | Agent | Deployments (agent / brute force TFLite / Qualia) |
|---|---|---|
| mnist_cnn_small → tiny | qualia int16, fits | **1** / 4 / 2 |
| mnist_cnn_medium → tiny | infeasible: ~220 KB of 8-bit weights > 64 KB flash | **0** / 4 / 3 |
| fashion_mlp_large → medium | tflite dynamic_int8, fits | **1** / 3 / 3 |

```bash
cp .env.example .env   # then paste a free key from https://console.groq.com/keys
python scripts/run_agent.py mnist_cnn_small tiny
python scripts/run_agent.py mnist_cnn_small tiny --provider ollama --llm qwen2.5:3b   # fully local
```

## Real chips: datasheet retrieval (RAG)

The target can also be a real microcontroller from [`configs/chips.yaml`](configs/chips.yaml): NXP MCX N947, i.MX RT1062, LPC55S66, LPC55S69, ST STM32L476RG, STM32WL33C8 and the RAKwireless RAK3172. The agent is only given the part number. It must search the datasheets, find the memory of that exact part, and declare it with `set_memory_budget` before it may deploy. The model then gets 50 % of the usable on-chip flash and 50 % of the SRAM (or a 4 MB external flash budget when the chip has no on-chip flash).

- **Retrieval**: `pdftotext -layout` keeps table rows on one line; pages are cut into overlapping chunks; BM25 (exact part numbers, units) and local `bge-small-en-v1.5` embeddings (paraphrases) are fused with reciprocal rank fusion.
- **Grounding check**: a memory declaration must quote, verbatim, a passage the search actually returned, and the declared sizes must appear in that quote; otherwise it is rejected and the agent has to try again.
- **Answer check**: a "feasible" answer must name a configuration the agent measured as fitting.

The datasheets contain traps, which is the point. Hand-checked ground truth is in `chips.yaml`, used only for scoring:

| Chip | Trap | Early runs |
|---|---|---|
| LPC55S69JBD100 | 640 KB flash, of which 10 KB reserved | ✅ 630 KB / 320 KB |
| MIMXRT1062 | no on-chip flash, 1 MB SRAM | ✅ detected, external flash budget |
| RAK3172 | the 512 KB external flash is only on the RAK3172-F | ✅ 256 KB / 64 KB |
| STM32WL33C8 | in "C8", C is the pin count and 8 the memory (64 KB / 16 KB) | ❌ both LLMs read 256 KB / 32 KB, even with forced part-number decoding |

The STM32WL33C8 failure is a misattribution, not a hallucination: the quoted table is real, but the wrong row is used. The grounding check cannot catch it, which is one of the things the evaluation will measure.

Datasheets are not versioned (copyright). Download them from the vendors' sites into `data/datasheets/` (file names in `chips.yaml`), then:

```bash
python scripts/build_index.py                       # ~2 min on CPU, ~2,000 chunks
python scripts/run_agent.py mnist_cnn_small LPC55S69JBD100
```

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
- [x] LangGraph agent: plan → act → observe loop with tool calling and memory of past attempts
- [x] Retrieval over MCU datasheets (RAG) to derive memory budgets from real chips, with grounding checks
- [x] Context management: 3 passages per search, datasheet passages pruned from the context once the memory budget is accepted
- [x] Bring your own model: a `.pt` (Qualia backend), a `.keras` (TFLite backend) or both, plus a `.npz` test set; the agent only uses the backends available for that model
- [ ] Evaluation harness: success rate, iterations, tokens and time vs. baseline, across LLMs
