"""Chat model factory. Both providers are free: Groq (hosted, free tier) or Ollama (local)."""

from __future__ import annotations

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from edge_deploy_agent.tools.deploy import ROOT

DEFAULT_MODELS = {"groq": "openai/gpt-oss-120b", "ollama": "qwen2.5:3b"}
OLLAMA_NUM_CTX = 8192


def get_llm(provider: str = "groq", model: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    model = model or DEFAULT_MODELS[provider]
    if provider == "groq":
        from langchain_groq import ChatGroq

        load_dotenv(ROOT / ".env")  # GROQ_API_KEY
        return ChatGroq(model=model, temperature=temperature, max_retries=3)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # Ollama's default context is a few thousand tokens; agent prompts with datasheet
        # passages need more. Larger contexts cost VRAM (KV cache).
        return ChatOllama(model=model, temperature=temperature, num_ctx=OLLAMA_NUM_CTX)
    raise ValueError(f"Unknown provider {provider!r}. Available: {sorted(DEFAULT_MODELS)}")
