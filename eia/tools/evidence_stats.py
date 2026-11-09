"""Numeric facts parsed from the *retrieved* documents only (never from gold labels). Agent 2 attaches these
to the Evidence Package so that a small local LLM compares numbers it has been handed instead of
doing arithmetic over prose. The tool reports ratios; it does not assign verdicts."""
from __future__ import annotations

import re
from statistics import mean
from typing import Optional

_ATT = re.compile(r"Actual attendance:\s*(\d+)")
_SIGN = re.compile(r"(\d+)\s+sign-ins")
_VAR = re.compile(r"\(([+-]?\d+(?:\.\d+)?)%\s*vs plan\)")
_SAT = re.compile(r"satisfaction:\s*(\d(?:\.\d+)?)/5")
_CPA = re.compile(r"Cost per attendee INR\s*([\d,]+)")


def _events(docs: list[dict]) -> dict[str, dict]:
    ev: dict[str, dict] = {}
    for d in docs:
        eid = d.get("event_id")
        if not eid:
            continue
        e = ev.setdefault(eid, {"event_type": d.get("event_type")})
        t = d["text"]
        for rx, key, cast in ((_ATT, "attendance", int), (_SIGN, "attendance", int), (_VAR, "variance_pct", float),
                              (_SAT, "satisfaction", float)):
            m = rx.search(t)
            if m and key not in e:
                e[key] = cast(m.group(1))
        m = _CPA.search(t)
        if m:
            e["cost_per_attendee"] = int(m.group(1).replace(",", ""))
    return ev


def attendance_stats(docs: list[dict], event_type: Optional[str], proposed: Optional[int]) -> dict:
    ev = [e for e in _events(docs).values() if e.get("event_type") == event_type and "attendance" in e]
    if not ev or not proposed:
        return {"n_events": len(ev), "note": "no same-type historical attendance found" if not ev else "proposed attendance missing"}
    hist = [e["attendance"] for e in ev]
    return {"n_events": len(hist), "historical_attendance": hist, "historical_max": max(hist),
            "historical_mean": round(mean(hist), 1), "proposed": proposed,
            "ratio_to_max": round(proposed / max(hist), 2), "ratio_to_mean": round(proposed / mean(hist), 2)}


def budget_stats(docs: list[dict], event_type: Optional[str], proposed_per_head: Optional[float]) -> dict:
    ev = [e for e in _events(docs).values() if e.get("event_type") == event_type and "cost_per_attendee" in e]
    if not ev or not proposed_per_head:
        return {"n_events": len(ev), "note": "no same-type historical cost-per-attendee found" if not ev else "proposed cost per head missing"}
    hist = [e["cost_per_attendee"] for e in ev]
    return {"n_events": len(hist), "historical_cost_per_attendee": hist, "historical_mean": round(mean(hist), 1),
            "proposed_per_attendee": round(proposed_per_head, 1), "ratio_to_mean": round(proposed_per_head / mean(hist), 2)}


def past_performance_stats(docs: list[dict]) -> dict:
    ev = _events(docs)
    var = [e["variance_pct"] for e in ev.values() if "variance_pct" in e]
    sat = [e["satisfaction"] for e in ev.values() if "satisfaction" in e]
    if not var and not sat:
        return {"n_events": 0, "note": "no past event reports found"}
    return {"n_events": len(ev), "budget_variance_pct": var,
            "max_overrun_pct": max(var) if var else None,
            "n_overruns_over_25pct": sum(v > 25 for v in var),
            "mean_satisfaction": round(mean(sat), 2) if sat else None,
            "min_satisfaction": min(sat) if sat else None}
