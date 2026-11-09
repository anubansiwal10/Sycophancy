"""Agent 2 - Evidence Retrieval (LLD: Retrieval Planner -> Query Embedder -> Semantic Search -> Ranker ->
Evidence Sufficient? -> [Query Refinement loop] -> Evidence Packager).

Vector search only (no graph). For every verifiable claim it issues a *supporting* query and an adversarial
*counter* query, so the Critical Reviewer always receives evidence that could contradict the proposal."""
from __future__ import annotations

from ..config import S
from ..llm import LLMFormatError, call_json
from ..state import GraphState, traced
from ..store import get_embedder, get_store
from ..tools.evidence_stats import attendance_stats, budget_stats, past_performance_stats
from ..tools.policy_facts import POLICY_QUERIES, applicable_policies, policy_facts

MIN_DOCS = {"Attendance": 1, "Budget": 1, "Past Performance": 2, "Policy": 1}
CAP = {"Attendance": 8, "Budget": 8, "Past Performance": 14, "Policy": 7}


# ------------------------------------------------------------------ planner ---------------------
def _queries_for(sp: dict, claim: dict, level: int) -> list[dict]:
    cid, club, et = sp.get("club_id"), sp.get("club_name") or "", sp.get("event_type") or ""
    t, c = claim["claim_type"], claim["claim_id"]
    if not cid and t != "Policy":  # club-specific evidence needs a known club; policies apply to everyone
        return []
    extra = 4 * level  # refinement widens recall
    if t == "Attendance":
        return [
            dict(claim_id=c, intent="supporting", text=f"{club} {et} attendance register sign-ins actual attendance expected", club_ids=[cid], event_type=et, doc_types=["Attendance Record", "Event Report"], k=6 + extra),
            dict(claim_id=c, intent="counter", text=f"{club} {et} lower than expected attendance no-show rate registered versus actual", club_ids=[cid], event_type=et, doc_types=["Attendance Record", "Committee Comment"] + (["Event Report"] if level else []), k=3 + extra),
        ]
    if t == "Budget":
        return [
            dict(claim_id=c, intent="supporting", text=f"{club} {et} budget statement actual spend cost per attendee", club_ids=[cid], event_type=et, doc_types=["Budget Sheet", "Event Report"], k=6 + extra),
            dict(claim_id=c, intent="counter", text=f"{club} {et} budget overrun variance exceeded planned spend", club_ids=[cid], event_type=et, doc_types=["Budget Sheet", "Committee Comment"], k=3 + extra),
        ]
    if t == "Past Performance":
        return [
            dict(claim_id=c, intent="supporting", text=f"{club} past events delivered on budget participant satisfaction", club_ids=[cid], event_type=None, doc_types=["Event Report", "Budget Sheet"], k=12 + extra),
            dict(claim_id=c, intent="counter", text=f"{club} budget overrun exceeded approved budget complaints late settlement", club_ids=[cid], event_type=None, doc_types=["Committee Comment", "Budget Sheet"], k=6 + extra),
        ]
    if t == "Policy":
        return [dict(claim_id=c, intent="supporting", text=POLICY_QUERIES[pid], club_ids=["ALL"], event_type=None, doc_types=["University Policy"], k=1, policy=pid)
                for pid in applicable_policies(sp)]
    return []


@traced("a2_retrieval_planner")
def retrieval_planner(state: GraphState) -> dict:
    sp, level = state["structured_proposal"], state.get("broaden_level", 0)
    plan = []
    for cl in sp["claims"]:
        if cl["verifiable"]:
            plan += _queries_for(sp, cl, level)
    if S.llm_planner and plan:  # optional: let Qwen rewrite query wording; falls back silently
        try:
            qs = [{"i": i, "claim": next(c["claim_text"] for c in sp["claims"] if c["claim_id"] == q["claim_id"]), "draft_query": q["text"]}
                  for i, q in enumerate(plan) if q["doc_types"] != ["University Policy"]]
            out = call_json("plan_queries", "Rewrite each draft retrieval query so it best finds evidence in club event reports, budget sheets and attendance registers. Keep club and event type names. Return {\"queries\":[{\"i\":int,\"query\":str}]}.",
                            {"queries": qs}, retries=0)
            for r in out.get("queries", []):
                plan[int(r["i"])]["text"] = str(r["query"])
        except (LLMFormatError, ValueError, KeyError, StopIteration, IndexError):
            pass
    return {"query_plan": plan, "gaps": [], "retrieval_attempts": state.get("retrieval_attempts", 0)}


# ------------------------------------------------------------------ search + rank --------------
@traced("a2_semantic_search")
def semantic_search(state: GraphState) -> dict:
    plan = state.get("query_plan", [])
    if not plan:
        return {"raw_hits": []}
    vecs = get_embedder().embed([q["text"] for q in plan])  # Query Embedder
    store = get_store()
    hits = []
    for q, v in zip(plan, vecs):
        for h in store.search(v, q["k"], club_ids=q["club_ids"], event_type=q["event_type"], doc_types=q["doc_types"]):
            hits.append({**h, "claim_id": q["claim_id"], "intent": q["intent"]})
    return {"raw_hits": hits}


def _in_scope(claim_type: str, doc: dict, sp: dict) -> bool:
    if claim_type in ("Attendance", "Budget"):
        return doc["club_id"] == sp.get("club_id") and doc["event_type"] == sp.get("event_type")
    if claim_type == "Past Performance":
        return doc["club_id"] == sp.get("club_id")
    return doc["doc_type"] == "University Policy"


@traced("a2_ranker")
def ranker(state: GraphState) -> dict:
    sp = state["structured_proposal"]
    ctype = {c["claim_id"]: c["claim_type"] for c in sp["claims"]}
    merged: dict[str, dict[str, dict]] = {}
    for h in state.get("raw_hits", []):
        d = merged.setdefault(h["claim_id"], {}).setdefault(h["doc_id"], {**h, "retrieved_as": set()})
        d["retrieved_as"].add(h["intent"])
        d["score"] = max(d["score"], h["score"])
    ranked = {}
    for cid, docs in merged.items():
        lst = []
        for d in docs.values():
            d["scope_match"] = _in_scope(ctype[cid], d, sp)
            d["retrieved_as"] = sorted(d["retrieved_as"])
            lst.append(d)
        lst.sort(key=lambda d: (not d["scope_match"], -d["score"], d["doc_date"]))
        ranked[cid] = lst[:CAP[ctype[cid]]]
    return {"ranked": ranked}


@traced("a2_sufficiency_check")
def sufficiency_check(state: GraphState) -> dict:
    sp, ranked = state["structured_proposal"], state.get("ranked", {})
    gaps = []
    for c in sp["claims"]:
        if not c["verifiable"]:
            continue
        n = sum(d["scope_match"] for d in ranked.get(c["claim_id"], []))
        if n < MIN_DOCS[c["claim_type"]]:
            gaps.append(c["claim_id"])
    return {"gaps": gaps}


def after_sufficiency(state: GraphState) -> str:
    if state.get("gaps") and state.get("retrieval_attempts", 0) < S.max_retrieval_attempts:
        return "a2_query_refinement"
    return "a2_evidence_packager"


@traced("a2_query_refinement")
def query_refinement(state: GraphState) -> dict:
    # Broaden recall (more results, extra doc types). Evidence outside the claim's scope is still flagged
    # scope_match=False and can never count as coverage, so refinement cannot manufacture support.
    return {"retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
            "broaden_level": state.get("broaden_level", 0) + 1}


@traced("a2_replan")
def replan(state: GraphState) -> dict:
    """Judge -> Agent 2 feedback loop ('Confidence / Evidence sufficient?' diamond in the HLD)."""
    return {"replans": state.get("replans", 0) + 1, "broaden_level": state.get("broaden_level", 0) + 1, "retrieval_attempts": 0}


# ------------------------------------------------------------------ packager -------------------
def _public(d: dict) -> dict:
    return {"doc_id": d["doc_id"], "doc_type": d["doc_type"], "title": d["title"], "event_id": d["event_id"],
            "event_type": d["event_type"], "doc_date": d["doc_date"], "score": round(d["score"], 3),
            "retrieved_as": d["retrieved_as"], "scope_match": d["scope_match"],
            "snippet": d["text"][:S.snippet_chars]}


@traced("a2_evidence_packager")
def evidence_packager(state: GraphState) -> dict:
    sp, ranked = state["structured_proposal"], state.get("ranked", {})
    claims_out, all_ids, n_ver, n_cov = [], [], 0, 0
    for c in sp["claims"]:
        docs = ranked.get(c["claim_id"], [])
        scope_docs = [d for d in docs if d["scope_match"]]
        entry = {"claim_id": c["claim_id"], "claim_type": c["claim_type"], "verifiable": c["verifiable"],
                 "supporting": [_public(d) for d in docs if "supporting" in d["retrieved_as"]],
                 "counter": [_public(d) for d in docs if d["retrieved_as"] == ["counter"]],
                 "stats": {}, "policy_facts": [], "evidence_available": bool(scope_docs), "gap_reason": ""}
        if c["verifiable"]:
            n_ver += 1
            if c["claim_type"] == "Attendance":
                entry["stats"] = attendance_stats(scope_docs, sp.get("event_type"), sp.get("expected_attendance"))
            elif c["claim_type"] == "Budget":
                entry["stats"] = budget_stats(scope_docs, sp.get("event_type"), sp.get("budget_per_attendee_inr"))
            elif c["claim_type"] == "Past Performance":
                entry["stats"] = past_performance_stats(scope_docs)
            elif c["claim_type"] == "Policy":
                # policy docs are titled "POL-02: Minimum Submission Lead Time" -> evaluate only retrieved policies
                entry["policy_facts"] = policy_facts(sp, {d["title"].split(":")[0].strip() for d in scope_docs})
            usable = scope_docs and (entry["stats"].get("n_events", 1) > 0 if c["claim_type"] != "Policy" else bool(entry["policy_facts"]))
            entry["evidence_available"] = bool(usable)
            if not entry["evidence_available"]:
                entry["gap_reason"] = ("club not found in knowledge base" if not sp.get("club_id") else
                                       f"no in-scope evidence retrieved for {c['claim_type']} (event type '{sp.get('event_type')}')")
            n_cov += entry["evidence_available"]
        all_ids += [d["doc_id"] for d in docs]
        claims_out.append(entry)
    uniq = sorted(set(all_ids))
    pkg = {"proposal_id": sp["proposal_id"], "club_id": sp.get("club_id"), "broaden_level": state.get("broaden_level", 0),
           "attempts": state.get("retrieval_attempts", 0), "claims": claims_out,
           "coverage": round(n_cov / n_ver, 3) if n_ver else 0.0, "n_verifiable": n_ver,
           "n_verifiable_no_evidence": n_ver - n_cov, "retrieved_doc_ids": uniq, "evidence_diversity": len(uniq)}
    return {"evidence_package": pkg}
