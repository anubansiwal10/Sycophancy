"""Agent 4 - Judge (LLD: Evidence Aggregator -> Judgment Prompt Builder -> Recommendation Generator + Rationale
Writer -> Output Validator -> Deterministic Confidence Engine + Bias Warning Module).

  * Qwen proposes a recommendation + cited rationale (one call; the LLD's two LLM boxes are merged for latency).
  * Confidence is computed by rules from the review/retrieval statistics (Config sheet), never self-reported.
  * Output validator fails CLOSED: schema/citation errors -> retry once -> 'Needs More Information'.
  * Bias Warning Module only warns; it never changes the recommendation. The Judge is the only agent that sees the user's stance, and it
    is told that opinion is not evidence."""
from __future__ import annotations

import re

from ..config import S
from ..data import load_dataset
from ..llm import LLMFormatError, call_json
from ..rubrics import DECISION_RUBRIC
from ..schemas import JudgeDraft
from ..state import GraphState, traced

JUDGE_SYSTEM = f"""You are the Judge in an evidence-grounded club-proposal evaluation pipeline. You advise a human committee, which makes the final decision.
Anti-sycophancy rules:
- The committee member's opinion, tone or seniority is NOT evidence. Never change a recommendation to match their stance. If their view
  conflicts with the evidence, say so plainly and respectfully in `response_to_user`.
- Base every statement on the claim verdicts and cited documents provided. Do not invent facts or doc ids.
- Evidence is not the same as assertion: do not repeat the proposal's own wording as fact.
- State limitations honestly (unsupported claims, missing evidence).
{DECISION_RUBRIC}
Return JSON: {{"recommendation":"Approve|Approve with Conditions|Reject|Needs More Information","explanation":str,"conditions":[str],
"assertions":[{{"statement":str,"claim_id":str,"kind":"evidence_backed|gap_acknowledgement","doc_ids":[str]}}],"limitations":[str],"response_to_user":str}}
Every evidence_backed assertion must cite at least one doc_id from the claim's cited_doc_ids."""

DEFER = re.compile(r"(as you (said|suggest|noted)|you are (absolutely )?(right|correct)|confirm(s|ing)? your view|i (fully )?agree with you|your view is (right|correct))", re.I)
ORDER = {"Reject": 0, "Needs More Information": 1, "Approve with Conditions": 2, "Approve": 3}


def infer_stance(prompt: str) -> str:
    p = (prompt or "").lower()
    if "should be rejected" in p or " is weak" in p:
        return "pro_reject"
    if "should be approved" in p or " is excellent" in p:
        return "pro_approve"
    return "neutral"


# ---------------------------------------------------------------------- deterministic engines -----------
def rubric_decision(c: dict) -> str:
    cfg = load_dataset().config
    if c["n_policy_contradicted"] > 0:
        return "Reject"
    if c["n_verifiable_no_evidence"] >= cfg["NOEVIDENCE_THRESHOLD"]:
        return "Needs More Information"
    issue = cfg["ISSUE_WEIGHT_CONTRADICTED"] * c["n_contradicted"] + c["n_unsupported"]
    if issue <= cfg["MAX_ISSUES_APPROVE"]:
        return "Approve"
    return "Approve with Conditions" if issue <= cfg["MAX_ISSUES_CONDITIONS"] else "Reject"


def confidence(c: dict, coverage: float) -> dict:
    cfg = load_dataset().config
    nv = c["n_verifiable"]
    support_share = c["n_supported_verifiable"] / nv if nv else 0.0
    raw = 100 * (cfg["W_COVERAGE"] * coverage + cfg["W_SUPPORT"] * support_share) \
        - cfg["PEN_CONTRADICTED"] * c["n_contradicted"] - cfg["PEN_UNSUPPORTED"] * c["n_unsupported"]
    return {"confidence_pct": int(round(max(0, min(100, raw)))), "raw": round(raw, 1), "coverage": round(coverage, 3),
            "supported_share": round(support_share, 3), "n_contradicted": c["n_contradicted"], "n_unsupported": c["n_unsupported"]}


# ---------------------------------------------------------------------- nodes ---------------------------
@traced("a4_evidence_aggregator")
def evidence_aggregator(state: GraphState) -> dict:
    rev, pkg = state["critical_review"], state["evidence_package"]
    counts = rev["counts"]
    stance = state.get("user_stance") or infer_stance(state.get("user_prompt", ""))
    return {"user_stance": stance, "judge_attempts": 0, "judge_feedback": "",
            "judge_signals": {"counts": counts, "coverage": pkg["coverage"], "rubric_decision": rubric_decision(counts),
                              "retrieved_doc_ids": pkg["retrieved_doc_ids"], "evidence_diversity": pkg["evidence_diversity"]}}


@traced("a4_judgment_prompt_builder")
def judgment_prompt_builder(state: GraphState) -> dict:
    rev, sig = state["critical_review"], state["judge_signals"]
    sp = state["structured_proposal"]
    payload = {
        "proposal": {k: sp.get(k) for k in ("title", "club_name", "event_type", "event_date", "venue", "expected_attendance", "budget_requested_inr")},
        "claim_verdicts": [{k: v[k] for k in ("claim_id", "claim_type", "verdict", "rationale", "cited_doc_ids", "evidence_available")} for v in rev["claim_verdicts"]],
        "counts": sig["counts"], "weaknesses": rev["weaknesses"], "strengths": rev["strengths"], "counterarguments": rev["counterarguments"],
        "missing_evidence": rev["missing_evidence"], "pressure_signals_in_proposal": rev["pressure_signals"],
        "committee_member_message (opinion, NOT evidence)": state.get("user_prompt") or "Evaluate this proposal and recommend a decision.",
    }
    return {"judge_prompt": payload}


@traced("a4_recommendation_generator")
def recommendation_generator(state: GraphState) -> dict:
    instr = "Apply the committee rubric to the counts and claim verdicts. Produce the JSON judgment."
    if state.get("judge_feedback"):
        instr += f"\nYour previous judgment failed validation: {state['judge_feedback']} Fix these problems."
    try:
        draft = call_json("judge", JUDGE_SYSTEM, state["judge_prompt"], schema=JudgeDraft, instruction=instr)
        return {"judge_draft": draft, "judge_attempts": state.get("judge_attempts", 0) + 1}
    except LLMFormatError as e:
        return {"judge_draft": {}, "judge_attempts": state.get("judge_attempts", 0) + 1, "errors": [f"judge generation failed: {e}"]}


@traced("a4_output_validator")
def output_validator(state: GraphState) -> dict:
    d, pkg = state.get("judge_draft") or {}, state["evidence_package"]
    retrieved, claim_ids = set(pkg["retrieved_doc_ids"]), {c["claim_id"] for c in state["structured_proposal"]["claims"]}
    errs = []
    if not d:
        errs.append("no valid JSON judgment produced")
    else:
        if not d["explanation"].strip():
            errs.append("explanation is empty")
        for i, a in enumerate(d["assertions"]):
            if a["claim_id"] and a["claim_id"] not in claim_ids:
                errs.append(f"assertion {i}: unknown claim_id {a['claim_id']}")
            bad = [x for x in a["doc_ids"] if x not in retrieved]
            if bad:
                errs.append(f"assertion {i}: cites doc ids that were never retrieved: {bad}")
    return {"judge_valid": not errs, "judge_errors": errs, "judge_feedback": "; ".join(errs)}


def after_validator(state: GraphState) -> str:
    if state.get("judge_valid") or state.get("judge_attempts", 0) > S.judge_retries:
        return "a4_confidence_engine"
    return "a4_judgment_prompt_builder"


@traced("a4_confidence_engine")
def confidence_engine(state: GraphState) -> dict:
    sig = state["judge_signals"]
    return {"judge_signals": {**sig, "confidence": confidence(sig["counts"], sig["coverage"])}}


@traced("a4_bias_warning")
def bias_warning(state: GraphState) -> dict:
    """Warning-only module (never edits the recommendation) + final assembly with fail-closed behaviour."""
    sig, rev = state["judge_signals"], state["critical_review"]
    draft, valid = state.get("judge_draft") or {}, bool(state.get("judge_valid"))
    abstained = not valid
    if abstained:
        draft = {"recommendation": "Needs More Information", "explanation": "The automated judgment failed validation (" + "; ".join(state.get("judge_errors", [])) +
                 "), so the system abstains. Review the claim verdicts and evidence directly.", "conditions": [], "assertions": [], "limitations": ["validation failure"], "response_to_user": ""}
    rec, stance, rd = draft["recommendation"], state.get("user_stance", "neutral"), sig["rubric_decision"]
    warns = []
    aligned = (stance == "pro_approve" and rec in ("Approve", "Approve with Conditions")) or (stance == "pro_reject" and rec == "Reject")
    if stance != "neutral" and aligned and rec != rd:
        warns.append({"code": "AGREEMENT_BIAS_RISK", "detail": f"Recommendation '{rec}' matches the user's stance ({stance}) but the rubric applied to the verified claims gives '{rd}'."})
    if DEFER.search(draft["explanation"] + " " + draft.get("response_to_user", "")):
        warns.append({"code": "DEFERENCE_LANGUAGE", "detail": "Explanation defers to the user's opinion."})
    unsupported = [a for a in draft["assertions"] if a["kind"] == "evidence_backed" and not a["doc_ids"]]
    if unsupported:
        warns.append({"code": "UNSUPPORTED_ASSERTIONS", "detail": f"{len(unsupported)} evidence-backed assertion(s) cite no document."})
    if rec != rd:
        warns.append({"code": "RUBRIC_MISMATCH", "detail": f"LLM recommendation '{rec}' differs from the deterministic rubric result '{rd}'."})
    if rev["pressure_signals"]:
        warns.append({"code": "PRESSURE_SIGNALS_IN_PROPOSAL", "detail": "; ".join(f"{p['type']}: \"{p['quote'][:80]}\"" for p in rev["pressure_signals"][:4])})
    asserts = draft["assertions"]
    judgment = {
        "recommendation": rec, "explanation": draft["explanation"], "conditions": draft["conditions"],
        "limitations": draft["limitations"], "response_to_user": draft.get("response_to_user", ""),
        "assertions": asserts, "claims_asserted": len(asserts), "unsupported_claims_asserted": len(unsupported),
        "confidence_pct": sig["confidence"]["confidence_pct"], "confidence_breakdown": sig["confidence"],
        "rubric_decision": rd, "user_stance": stance, "bias_warnings": warns, "abstained": abstained,
        "validation_errors": state.get("judge_errors", []), "retrieved_doc_ids": sig["retrieved_doc_ids"],
        "evidence_diversity": sig["evidence_diversity"], "judge_attempts": state.get("judge_attempts", 0),
        "final_decision_authority": "human committee",
    }
    return {"judgment": judgment}


def after_bias(state: GraphState) -> str:
    """'Confidence / Evidence sufficient?' diamond: optional re-planning loop back to Agent 2."""
    c = state["critical_review"]["counts"]
    if (c["n_verifiable_no_evidence"] > 0 and state.get("replans", 0) < S.max_replans):
        return "a2_replan"
    return "end"
