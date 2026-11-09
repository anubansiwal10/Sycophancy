"""Central configuration. Every knob is an environment variable so nothing in agent code changes
when you swap models, embeddings or switch the Agent-1 bypass on/off.

    EIA_LLM_MODEL=qwen2.5:14b python run.py ...
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # ---- LLM (local Qwen through Ollama) -------------------------------------------------
    llm_provider: str = field(default_factory=lambda: _env("EIA_LLM_PROVIDER", "ollama"))  # ollama | openai | fake
    llm_model: str = field(default_factory=lambda: _env("EIA_LLM_MODEL", "qwen2.5:7b"))   # <- the ONE line to swap
    # hosted option (provider "openai" = any OpenAI-compatible API: Groq, Gemini, OpenRouter, ...)
    api_base_url: str = field(default_factory=lambda: _env("EIA_API_BASE_URL", "https://api.groq.com/openai/v1"))
    api_key: str = field(default_factory=lambda: _env("EIA_API_KEY", ""))
    api_reasoning_effort: str = field(default_factory=lambda: _env("EIA_API_REASONING_EFFORT", ""))  # e.g. "none" stops Qwen3 spending output quota on hidden thinking (only if the model supports it)
    terse: bool = field(default_factory=lambda: _flag("EIA_TERSE", True))   # brevity instruction on every prompt = fewer output tokens
    log_tokens: bool = field(default_factory=lambda: _flag("EIA_LOG_TOKENS", True))  # one line per API call: task, output tokens, seconds
    max_tokens: dict = field(default_factory=lambda: {"analyze": 900, "plan_queries": 300, "review_claims": 650, "challenge": 450, "judge": 700, "baseline": 350})
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_temperature: float = field(default_factory=lambda: float(_env("EIA_LLM_TEMPERATURE", "0")))
    llm_num_ctx: int = field(default_factory=lambda: int(_env("EIA_LLM_NUM_CTX", "8192")))
    llm_timeout: int = field(default_factory=lambda: int(_env("EIA_LLM_TIMEOUT", "900")))
    llm_think: str = field(default_factory=lambda: _env("EIA_LLM_THINK", ""))  # "", "true" or "false" (qwen3 thinking)
    llm_json_retries: int = field(default_factory=lambda: int(_env("EIA_LLM_JSON_RETRIES", "2")))

    # ---- Agent 1 bypass (testing only) ----------------------------------------------------
    use_cached_analysis: bool = field(default_factory=lambda: _flag("EIA_USE_CACHED_ANALYSIS", False))
    cached_dir: Path = field(default_factory=lambda: Path(_env("EIA_CACHED_DIR", str(ROOT / "data" / "cached_analysis"))))

    # ---- Data + retrieval (vector search only, NO graph database) ---------------------------
    dataset_path: Path = field(default_factory=lambda: Path(_env("EIA_DATASET", str(ROOT / "data" / "club_proposal_sycophancy_dataset.xlsx"))))
    vector_backend: str = field(default_factory=lambda: _env("EIA_VECTOR_BACKEND", "chroma"))  # chroma | qdrant (embedded, no server)
    vector_dir: Path = field(default_factory=lambda: Path(_env("EIA_VECTOR_DIR", str(ROOT / ".vector_store"))))
    embed_provider: str = field(default_factory=lambda: _env("EIA_EMBED_PROVIDER", "ollama"))  # ollama | hash (offline tests)
    embed_model: str = field(default_factory=lambda: _env("EIA_EMBED_MODEL", "nomic-embed-text"))

    # ---- Agent behaviour ----------------------------------------------------------------------
    llm_planner: bool = field(default_factory=lambda: _flag("EIA_LLM_PLANNER", False))  # Agent 2 query rewriting with Qwen
    max_retrieval_attempts: int = field(default_factory=lambda: int(_env("EIA_MAX_RETRIEVAL_ATTEMPTS", "2")))
    max_replans: int = field(default_factory=lambda: int(_env("EIA_MAX_REPLANS", "0")))  # Judge -> Agent 2 feedback loop
    judge_retries: int = field(default_factory=lambda: int(_env("EIA_JUDGE_RETRIES", "1")))
    snippet_chars: int = field(default_factory=lambda: int(_env("EIA_SNIPPET_CHARS", "240")))
    prompt_docs_per_claim: int = field(default_factory=lambda: int(_env("EIA_PROMPT_DOCS_PER_CLAIM", "6")))

    runs_dir: Path = field(default_factory=lambda: Path(_env("EIA_RUNS_DIR", str(ROOT / "runs"))))


S = Settings()
