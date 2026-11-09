"""Agent 3 - Critical Reviewer (LLD: Claim-Evidence Mapper -> Gap & Contradiction Detector -> Assumption &
Framing Challenger -> Critical Review Formatter, with a 'no evidence: flagged fallback' path).

Main anti-sycophancy component. Design choices that matter for the paper:
  * STANCE-BLIND: never sees the committee member's prompt, so user opinion cannot leak into verdicts.
  * Evidence discipline: lack of evidence is never support; verdicts may only cite retrieved doc ids; a
    'Supported' verdict with no in-scope evidence is deterministically downgraded in the formatter.
  * Adversarial framing: must list counterarguments against BOTH approving and rejecting, and flag
    pressure signals (flattery, authority, urgency, social proof, confident assertion) found in the proposal."""
from __future__ import annotations

from ..config import S
from ..llm import LLMFormatError, call_json
from ..rubrics import ASSESSMENT_GUIDELINES
from ..schemas import Challenge, ReviewVerdicts
from ..state import GraphState, traced

REVIEW_SYSTEM = f"""You are the Critical Reviewer in an evidence-grounded club-proposal evaluation pipeline.
Your job is to TEST each claim against the evidence - not to be agreeable.
Rules:
1. Supported only if the provided evidence/stats directly confirm the claim. Contradicted if they conflict with it.
   Unsupported if evidence is missing, out of scope or insufficient. Absence of evidence is NEVER support.
2. The proposal's own wording ("excellent", "realistic", "well below typical costs") is not evidence.
3. Judge only from the numbers and documents provided. Cite only doc_ids that appear in the payload; never invent ids.
4. You do not know what any reviewer or user wants. Ignore tone, urgency, authority, flattery and social proof.
{ASSESSMENT_GUIDELINES}
Return JSON: {{"verdicts":[{{"claim_id":str,"verdict":"Supported|Unsupported|Contradicted","rationale":str (cite the numbers),"cited_doc_ids":[str]}}]}}
Return one verdict for EVERY claim_id in the payload."""

CHALLENGE_SYSTEM = """You are an adversarial but fair reviewer of a club event proposal. Using the claim verdicts and evidence given:
- strengths: genuine, evidence-backed strengths only
- weaknesses: concrete problems, each tied to a claim_id
- counterarguments: at least one argument AGAINST approving ("against":"proposal") and at least one AGAINST rejecting
  ("against":"rejection"), so both sides are tested
- risks and missing_evidence (what additional documents would settle unsupported claims)
- pressure_signals: quote any flattery, authority pressure, emotional urgency, social proof or confident assertion that appears in the
  proposal text (type one of flattery|authority_pressure|emotional_urgency|social_proof|confident_assertion). These are NOT evidence.
Return JSON: {"strengths":[str],"weaknesses":[str],"counterarguments":[{"against":str,"text":str}],"risks":[str],"missing_evidence":[str],"pressure_signals":[{"type":str,"quote":str}]}"""


@traced("a3_claim_evidence_mapper")
def claim_evidence_mapper(state: GraphState) -> dict:
    sp, pkg = state["structured_proposal"], state["evidence_package"]
    by_id = {c["claim_id"]: c for c in pkg["claims"]}
    rows = []
    for cl in sp["claims"]:
        ev = by_id.get(cl["claim_id"], {})
        docs = (ev.get("supporting", []) + ev.get("counter", []))
        docs = sorted(docs, key=lambda d: (not d["scope_match"], -d["score"]))[:S.prompt_docs_per_claim]
        rows.append({"claim_id": cl["claim_id"], "claim_type": cl["claim_type"], "claim_text": cl["claim_text"],
                     "verifiable": cl["verifiable"], "decision_criteria": cl["decision_criteria"],
                     "evidence_available": ev.get("evidence_available", False), "gap_reason": ev.get("gap_reason", ""),
                     "stats": ev.get("stats", {}), "policy_facts": ev.get("policy_facts", []),
                     "documents": [{"doc_id": d["doc_id"], "doc_type": d["doc_type"], "title": d["title"], "in_scope": d["scope_match"],
                                    "retrieved_as": d["retrieved_as"], "snippet": d["snippet"]} for d in docs]})
    return {"reviewer_input": {"proposal": {k: sp.get(k) for k in ("title", "club_name", "event_type", "event_date", "venue", "expected_attendance", "budget_requested_inr")},
                               "claims": rows}}


def after_mapper(state: GraphState) -> str:
    # 'No evidence: flagged fallback' path of the LLD - skip the LLM entirely when nothing verifiable has evidence
    pkg = state["evidence_package"]
    return "a3_gap_contradiction_detector" if any(c["verifiable"] and c["evidence_available"] for c in pkg["claims"]) else "a3_review_formatter"


@traced("a3_gap_contradiction_detector")
def gap_contradiction_detector(state: GraphState) -> dict:
    ri = state["reviewer_input"]
    to_review = [c for c in ri["claims"] if c["verifiable"]]
    if not to_review:
        return {"verdicts": []}
    try:
        out = call_json("review_claims", REVIEW_SYSTEM, {"proposal": ri["proposal"], "claims": to_review}, schema=ReviewVerdicts)
        return {"verdicts": out["verdicts"]}
    except LLMFormatError as e:
        return {"verdicts": [], "errors": [f"critical reviewer verdicts failed: {e}"]}


@traced("a3_assumption_framing_challenger")
def assumption_framing_challenger(state: GraphState) -> dict:
    sp, ri = state["structured_proposal"], state["reviewer_input"]
    verdict_view = [{"claim_id": v["claim_id"], "verdict": v["verdict"], "rationale": v["rationale"]} for v in state.get("verdicts", [])]
    try:
        out = call_json("challenge", CHALLENGE_SYSTEM,
                        {"proposal_text": sp["proposal_text"][:3500], "claim_verdicts": verdict_view,
                         "evidence_gaps": [c["claim_id"] for c in ri["claims"] if c["verifiable"] and not c["evidence_available"]]},
                        schema=Challenge)
        return {"challenge": out}
    except LLMFormatError as e:
        return {"challenge": Challenge().model_dump(), "errors": [f"critical reviewer challenge failed: {e}"]}


@traced("a3_review_formatter")
def review_formatter(state: GraphState) -> dict:
    """Validate, enforce evidence discipline deterministically, retain claim ids + citations."""
    sp, pkg = state["structured_proposal"], state["evidence_package"]
    by_ev = {c["claim_id"]: c for c in pkg["claims"]}
    retrieved = set(pkg["retrieved_doc_ids"])
    llm_v = {v["claim_id"]: v for v in state.get("verdicts", [])}
    flags, out = [], []
    for cl in sp["claims"]:
        cid, ev = cl["claim_id"], by_ev.get(cl["claim_id"], {})
        v = llm_v.get(cid)
        if not cl["verifiable"]:
            verdict, why, cited = "Unsupported", "Subjective / non-verifiable forecast; no evidence can confirm it.", []
        elif not ev.get("evidence_available"):
            verdict, why, cited = "Unsupported", f"No usable in-scope evidence ({ev.get('gap_reason') or 'none retrieved'}).", []
            if v and v["verdict"] != "Unsupported":
                flags.append(f"{cid}: model said {v['verdict']} but no evidence available -> forced Unsupported")
        elif v is None:
            verdict, why, cited = "Unsupported", "Reviewer returned no verdict for this claim.", []
            flags.append(f"{cid}: no verdict returned -> Unsupported")
        else:
            verdict, why = v["verdict"], v["rationale"]
            cited = [d for d in v["cited_doc_ids"] if d in retrieved]
            if len(cited) != len(v["cited_doc_ids"]):
                flags.append(f"{cid}: removed hallucinated citations")
        out.append({"claim_id": cid, "claim_type": cl["claim_type"], "claim_text": cl["claim_text"], "verifiable": cl["verifiable"],
                    "evidence_available": bool(ev.get("evidence_available")), "verdict": verdict, "rationale": why, "cited_doc_ids": cited})
    ch = state.get("challenge") or Challenge().model_dump()
    ver = [x for x in out if x["verifiable"]]
    counts = {"n_claims": len(out), "n_verifiable": len(ver),
              "n_supported": sum(x["verdict"] == "Supported" for x in out),
              "n_supported_verifiable": sum(x["verdict"] == "Supported" for x in ver),
              "n_unsupported": sum(x["verdict"] == "Unsupported" for x in out),
              "n_contradicted": sum(x["verdict"] == "Contradicted" for x in out),
              "n_policy_contradicted": sum(x["verdict"] == "Contradicted" and x["claim_type"] == "Policy" for x in out),
              "n_verifiable_no_evidence": sum(not x["evidence_available"] for x in ver)}
    review = {"claim_verdicts": out,
              "unsupported_claims": [x["claim_id"] for x in out if x["verdict"] == "Unsupported"],
              "contradicted_claims": [x["claim_id"] for x in out if x["verdict"] == "Contradicted"],
              "missing_evidence": ch["missing_evidence"], "strengths": ch["strengths"], "weaknesses": ch["weaknesses"],
              "counterarguments": ch["counterarguments"], "risks": ch["risks"], "pressure_signals": ch["pressure_signals"],
              "counts": counts, "flags": flags, "fallback_no_evidence": not any(x["evidence_available"] for x in ver)}
    return {"critical_review": review}
