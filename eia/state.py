"""Shared LangGraph state + per-node tracing (feeds the latency tables in the paper)."""
from __future__ import annotations

import functools
import operator
import time
from typing import Annotated, Any, TypedDict

from . import llm


class GraphState(TypedDict, total=False):
    # ---- inputs
    proposal_text: str
    proposal_path: str
    proposal_id: str
    analysis_json_path: str      # optional: ready-made Agent-1 JSON (bypass for any proposal, not only benchmark ids)
    proposal_form: dict          # submission-form fields (dates, venue capacity, approvals ...)
    user_prompt: str             # what the committee member typed (may carry a stance = sycophancy pressure)
    user_stance: str             # neutral | pro_approve | pro_reject (optional, inferred if absent)
    # ---- agent 1
    parsed_text: str
    analysis_attempts: int
    analysis_feedback: str
    analysis_instruction: str
    candidate_json: dict
    analysis_ok: bool
    analysis_source: str         # llm | cached
    structured_proposal: dict
    # ---- agent 2
    broaden_level: int
    retrieval_attempts: int
    replans: int
    query_plan: list
    raw_hits: list
    ranked: dict
    gaps: list
    evidence_package: dict
    # ---- agent 3
    reviewer_input: dict
    verdicts: list
    challenge: dict
    critical_review: dict
    # ---- agent 4
    judge_signals: dict
    judge_prompt: dict
    judge_attempts: int
    judge_feedback: str
    judge_draft: dict
    judge_valid: bool
    judge_errors: list
    judgment: dict
    # ---- bookkeeping
    trace: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]


def traced(name: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state: GraphState) -> dict:
            s0, t0 = llm.stats(), time.perf_counter()
            out = dict(fn(state) or {})
            s1 = llm.stats()
            out["trace"] = [{"node": name, "seconds": round(time.perf_counter() - t0, 3),
                             "llm_calls": s1["calls"] - s0["calls"],
                             "llm_seconds": round(s1["seconds"] - s0["seconds"], 3)}]
            return out
        return wrapper
    return deco
