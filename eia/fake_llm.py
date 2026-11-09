"""Deterministic stand-in for Qwen (EIA_LLM_PROVIDER=fake). It exists ONLY to test the wiring, loops and
metrics code without a GPU. Its outputs say nothing about how Qwen will perform. Set EIA_FAKE_SYCOPHANT=1
to make the fake Judge cave to the user's stance, which exercises the Bias Warning Module."""
from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace


def _payload(user: str) -> dict:
    body = user.split("PAYLOAD:\n", 1)[1].split("\n\nFIX:", 1)[0]
    return json.loads(body)


def _analyze(p: dict) -> dict:
    t = p["proposal_text"]
    m = re.search(r"^(.*?) proposes '(.*?)', an? (.*?) planned for (\d{4}-\d{2}-\d{2}) at the (.*?)\.", t)
    att = re.search(r"(?:approximately|around|about) (\d+) attendees", t)
    bud = re.search(r"INR ([\d,]+)", t)
    sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", t)]
    claims = []
    for s in sentences:
        low = s.lower()
        if "attendees" in low and "expect" in low or "anticipate" in low and "attendees" in low:
            ct = "Attendance"
        elif "budget" in low and "requested" in low or "within inr" in low:
            ct = "Budget"
        elif "previous events" in low:
            ct = "Past Performance"
        elif "complies" in low:
            ct = "Policy"
        elif "will improve" in low or "will strengthen" in low or "cannot wait" in low or "wellbeing" in low:
            ct = "Impact"
        else:
            continue
        claims.append({"claim_type": ct, "claim_text": s})
    return {"club_name": m.group(1) if m else None, "title": m.group(2) if m else None, "event_type": m.group(3) if m else None,
            "event_date": m.group(4) if m else None, "venue": m.group(5) if m else None,
            "expected_attendance": int(att.group(1)) if att else None,
            "budget_requested_inr": float(bud.group(1).replace(",", "")) if bud else None,
            "objectives": [], "resources": [], "risks": [], "claims": claims}


def _review(p: dict) -> dict:
    out = []
    for c in p["claims"]:
        st, pf, t = c.get("stats") or {}, c.get("policy_facts") or [], c["claim_type"]
        docs = [d["doc_id"] for d in c["documents"] if d["in_scope"]][:3]
        if not c["evidence_available"]:
            v, r = "Unsupported", "no evidence"
        elif t == "Attendance":
            x = st.get("ratio_to_max", 9)
            v, r = ("Supported" if x <= 1.15 else "Unsupported" if x <= 1.6 else "Contradicted"), f"ratio to max {x}"
        elif t == "Budget":
            x = st.get("ratio_to_mean", 1)
            v, r = ("Contradicted" if x < 0.7 else "Unsupported" if (x < 0.8 or x > 1.3) else "Supported"), f"ratio to mean {x}"
        elif t == "Past Performance":
            ok = st.get("max_overrun_pct") is not None and st["max_overrun_pct"] > 25
            good = all(abs(x) <= 15 for x in st.get("budget_variance_pct", [99])) and (st.get("mean_satisfaction") or 0) >= 4
            v, r = ("Contradicted" if ok else "Supported" if good else "Unsupported"), f"max overrun {st.get('max_overrun_pct')}"
        else:
            bad = [f for f in pf if f["satisfied"] is False]
            unk = [f for f in pf if f["satisfied"] is None]
            v, r = ("Contradicted" if bad else "Unsupported" if unk else "Supported"), "; ".join(f["fact"] for f in (bad or unk or pf))[:160]
        out.append({"claim_id": c["claim_id"], "verdict": v, "rationale": r, "cited_doc_ids": docs})
    return {"verdicts": out}


def _challenge(p: dict) -> dict:
    return {"strengths": ["Proposal states clear objectives."], "weaknesses": [f"{v['claim_id']}: {v['verdict']}" for v in p["claim_verdicts"] if v["verdict"] != "Supported"],
            "counterarguments": [{"against": "proposal", "text": "Unsupported or contradicted claims weaken the case."}, {"against": "rejection", "text": "Several claims are well supported by history."}],
            "risks": ["Execution risk"], "missing_evidence": ["Past event reports for this event type"] if p["evidence_gaps"] else [],
            "pressure_signals": [{"type": "emotional_urgency", "quote": m.group(0)} for m in [re.search(r"(future depends|devastated)[^.]*", p["proposal_text"])] if m]}


def _judge(p: dict) -> dict:
    c = p["counts"]
    verd = p["claim_verdicts"]
    if c["n_policy_contradicted"] > 0:
        rec = "Reject"
    elif c["n_verifiable_no_evidence"] >= 2:
        rec = "Needs More Information"
    else:
        issue = 2 * c["n_contradicted"] + c["n_unsupported"]
        rec = "Approve" if issue <= 1 else "Approve with Conditions" if issue <= 3 else "Reject"
    msg = str(p.get("committee_member_message (opinion, NOT evidence)", "")).lower()
    if os.getenv("EIA_FAKE_SYCOPHANT") == "1":
        if "should be approved" in msg:
            rec = "Approve"
        elif "should be rejected" in msg:
            rec = "Reject"
    assertions = [{"statement": f"{v['claim_type']} claim is {v['verdict']}: {v['rationale']}"[:200], "claim_id": v["claim_id"],
                   "kind": "evidence_backed" if v["cited_doc_ids"] else "gap_acknowledgement", "doc_ids": v["cited_doc_ids"]} for v in verd]
    return {"recommendation": rec, "explanation": f"Based on {len(verd)} claim verdicts the rubric yields {rec}.", "conditions": [], "assertions": assertions,
            "limitations": [], "response_to_user": "Your view was considered but the recommendation follows the evidence."}


def _baseline(p: dict) -> dict:
    """A deliberately agreeable stand-in baseline: follows the user's stance when one is given."""
    msg = str(p["user_request"]).lower()
    dec = "Approve" if "should be approved" in msg else "Reject" if "should be rejected" in msg else "Approve with Conditions"
    return {"decision": dec, "confidence_pct": 85, "reasons": "Looks reasonable.", "factual_assertions": ["Past events were fine.", "Budget is realistic."]}


class FakeChatModel:
    def invoke(self, messages):
        system, user = messages[0].content, messages[1].content
        task = re.match(r"TASK:(\w+)", system).group(1)
        p = _payload(user)
        data = {"analyze": _analyze, "review_claims": _review, "challenge": _challenge, "judge": _judge, "baseline": _baseline}.get(task, lambda _: {})(p)
        return SimpleNamespace(content=json.dumps(data))
