"""Client-side throttle + 429 retry for hosted APIs with small per-minute quotas (e.g. Groq on-demand:
'output tokens per minute (OTPM): Limit 1000').

Why the original run failed: the provider rejects any single request whose *expected* output (max_tokens, or the
model default) is larger than the per-minute output limit, and rejects bursts that exceed it. So we
  1. always send an explicit max_tokens that fits under the limit,
  2. keep a 60 s sliding window of tokens used/reserved and WAIT before a call instead of failing,
  3. on a 429 anyway, sleep for the time the provider asks for ("try again in 12.3s" / Retry-After) and retry.
Limits are env vars (EIA_OTPM_LIMIT, EIA_TPM_LIMIT, EIA_RPM_LIMIT); 0 disables a limit."""
from __future__ import annotations

import os
import re
import threading
import time

OTPM = int(os.getenv("EIA_OTPM_LIMIT", "1000"))
TPM = int(os.getenv("EIA_TPM_LIMIT", "0"))
RPM = int(os.getenv("EIA_RPM_LIMIT", "0"))
SAFETY = float(os.getenv("EIA_LIMIT_SAFETY", "0.9"))   # use at most 90% of each quota
RESERVE = float(os.getenv("EIA_RESERVE_FRACTION", "0.6"))  # a call is admitted if this share of its max_tokens fits the window
MAX_RETRIES = int(os.getenv("EIA_429_RETRIES", "8"))

_lock = threading.Lock()
_events: list[dict] = []   # {"t": float, "out": int, "inp": int, "open": bool}


def est_tokens(text: str) -> int:
    return int(len(text) / 3.2) + 1


def _prune(now: float):
    _events[:] = [e for e in _events if now - e["t"] < 60]


def _wait_needed(now: float, out_req: int, in_req: int) -> float:
    _prune(now)
    waits = [0.0]
    for limit, key, need in ((OTPM, "out", out_req), (TPM, "tot", out_req + in_req), (RPM, "req", 1)):
        if not limit:
            continue
        cap = limit * SAFETY
        if need > cap:
            raise ValueError(f"a single request needs {need} tokens but the {key} limit is {limit}; lower max_tokens or the prompt")
        used = lambda e: e["out"] if key == "out" else (e["out"] + e["inp"] if key == "tot" else 1)
        tot = sum(used(e) for e in _events)
        if tot + need > cap:
            run, w = tot, 0.0
            for e in sorted(_events, key=lambda e: e["t"]):   # wait until enough old usage ages out of the window
                run -= used(e)
                w = e["t"] + 60 - now
                if run + need <= cap:
                    break
            waits.append(w)
    return max(waits)


def _retry_after(exc: Exception) -> float:
    msg = str(exc)
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s", msg, re.I)
    if m:
        return float(m.group(1) or 0) * 60 + float(m.group(2)) + 0.5
    hdrs = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        return float(hdrs.get("retry-after")) + 0.5
    except (TypeError, ValueError):
        return 0.0


def is_rate_limit(exc: Exception) -> bool:
    return "RateLimit" in type(exc).__name__ or getattr(exc, "status_code", None) == 429 or "429" in str(exc)[:80]


def limited_call(fn, max_tokens: int, prompt_text: str, log=print):
    """Run fn() under the quotas. Returns (result, output_tokens_used or None)."""
    inp = est_tokens(prompt_text)
    if OTPM and max_tokens > OTPM:  # the exact failure seen on Groq: one request may not ask for more than the whole per-minute quota
        raise ValueError(f"max_tokens={max_tokens} is above the output-tokens-per-minute limit ({OTPM}); lower max_tokens for this task in config.py")
    reserve = max(1, int(max_tokens * RESERVE))
    for attempt in range(MAX_RETRIES + 1):
        while True:
            with _lock:
                now = time.time()
                w = _wait_needed(now, reserve, inp)
                if w <= 0:
                    ev = {"t": now, "out": reserve, "inp": inp}   # reserve a typical output while in flight
                    _events.append(ev)
                    break
            time.sleep(min(w, 5.0) + 0.05)
        try:
            res = fn()
            usage = getattr(res, "usage_metadata", None) or {}
            with _lock:   # replace the reservation by what was really used
                ev["out"] = int(usage.get("output_tokens", ev["out"]))
                ev["inp"] = int(usage.get("input_tokens", ev["inp"]))
            return res
        except Exception as e:  # noqa: BLE001
            if not is_rate_limit(e) or attempt == MAX_RETRIES:
                with _lock:
                    ev["out"] = 0
                raise
            wait = max(_retry_after(e), 2.0 * (attempt + 1))
            log(f"  [rate limit] provider said wait; sleeping {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            with _lock:
                ev["t"] = time.time() + wait   # count the failed attempt as spent
