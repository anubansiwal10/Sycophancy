"""LLM access. One function (`call_json`) is used by every agent, so swapping Qwen for anything else
is a config change (EIA_LLM_MODEL / EIA_LLM_PROVIDER), never an agent change."""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Optional, Type

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from .config import S

_STATS = {"calls": 0, "seconds": 0.0, "out_tokens": 0}
TERSE = ("\nBe concise: every free-text field under 40 words, at most 4 list items per list, no preamble, no repetition of the input. "
         "Return the JSON object only.")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLMFormatError(RuntimeError):
    pass


def stats() -> dict:
    return dict(_STATS)


@lru_cache(maxsize=1)
def get_llm():
    if S.llm_provider == "fake":
        from .fake_llm import FakeChatModel
        return FakeChatModel()
    if S.llm_provider == "openai":  # hosted model through any OpenAI-compatible endpoint (nothing runs on the laptop)
        from langchain_openai import ChatOpenAI
        if not S.api_key:
            raise RuntimeError("set EIA_API_KEY (and EIA_API_BASE_URL / EIA_LLM_MODEL) to use EIA_LLM_PROVIDER=openai")
        kw = dict(model=S.llm_model, base_url=S.api_base_url, api_key=S.api_key, temperature=S.llm_temperature,
                  timeout=S.llm_timeout, max_retries=0,  # 429s are handled by ratelimit.limited_call (waits as long as the provider asks)
                  model_kwargs={"response_format": {"type": "json_object"}})
        if S.api_reasoning_effort:
            kw["extra_body"] = {"reasoning_effort": S.api_reasoning_effort}
        return ChatOpenAI(**kw)
    from langchain_ollama import ChatOllama
    kwargs = dict(model=S.llm_model, base_url=S.ollama_base_url, temperature=S.llm_temperature,
                  num_ctx=S.llm_num_ctx, format="json", client_kwargs={"timeout": S.llm_timeout})
    if S.llm_think.lower() in {"true", "false"}:
        kwargs["reasoning"] = S.llm_think.lower() == "true"
    return ChatOllama(**kwargs)


def extract_json(text: str) -> dict:
    text = _THINK.sub("", text).strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            return json.loads(text[a:b + 1])
        raise


def call_json(task: str, system: str, payload: dict, schema: Optional[Type[BaseModel]] = None,
              instruction: str = "", retries: Optional[int] = None) -> dict:
    """Invoke the model and return validated JSON. Malformed output triggers a re-prompt that quotes the
    error (this is the 'malformed output: re-prompt' loop in the Agent-1 LLD, reused by Agents 2-4)."""
    retries = S.llm_json_retries if retries is None else retries
    sys_msg = f"TASK:{task}\n{system}" + (TERSE if S.terse else "")
    body = f"{instruction}\n\nPAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=1, default=str)}"
    fix = ""
    last_err = ""
    for attempt in range(retries + 1):
        t = time.perf_counter()
        msgs = [SystemMessage(content=sys_msg), HumanMessage(content=body + fix)]
        if S.llm_provider == "openai":
            from .ratelimit import limited_call
            mt = S.max_tokens.get(task, 600)
            bound = get_llm().bind(max_tokens=mt)  # explicit output cap: the provider rejects requests whose expected output exceeds the per-minute quota
            resp = limited_call(lambda: bound.invoke(msgs), mt, sys_msg + body + fix)
            used = int((getattr(resp, "usage_metadata", None) or {}).get("output_tokens", 0))
            _STATS["out_tokens"] += used
            if S.log_tokens:
                print(f"  [llm] {task}: {used} output tokens (cap {mt}) in {time.perf_counter() - t:.1f}s", flush=True)
        else:
            resp = get_llm().invoke(msgs)
        _STATS["calls"] += 1
        _STATS["seconds"] += time.perf_counter() - t
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        try:
            data = extract_json(raw)
            if schema is not None:
                data = schema.model_validate(data).model_dump()
            return data
        except (ValueError, ValidationError) as e:
            last_err = str(e)[:600]
            fix = (f"\n\nFIX:\nYour previous reply was invalid ({last_err}). "
                   "Return ONLY one valid JSON object that follows the contract exactly, and keep it SHORT (the output may have been cut off).")
    raise LLMFormatError(f"{task}: model output invalid after {retries + 1} attempts: {last_err}")
