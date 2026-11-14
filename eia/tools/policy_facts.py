"""Deterministic policy arithmetic (dates, thresholds, capacity). The tool reports FACTS and a
satisfied/violated flag for each applicable university policy; the Critical Reviewer (LLM) still has to
turn them into claim verdicts. A rule is only evaluated if its policy document was actually retrieved
from the knowledge base. Thresholds mirror POL-01..POL-07 in the KnowledgeBase sheet."""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

POLICY_QUERIES = {
    "POL-01": "event budget ceiling Dean of Student Affairs written approval total budget",
    "POL-02": "minimum submission lead time proposals submitted days before event date",
    "POL-03": "examination blackout periods no club events",
    "POL-04": "venue capacity limits expected attendance must not exceed capacity",
    "POL-05": "external speakers or judges prior written approval faculty advisor",
    "POL-06": "safety and security plan crowd management first aid more than 200 attendees",
    "POL-07": "fundraising charity collection registered Finance Office deposited",
}
CAPACITY = {"Lab Block": 60, "Seminar Hall": 150, "Main Auditorium": 500, "Open Air Theatre": 800}


def _yes(v) -> Optional[bool]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"yes", "y", "true"}:
        return True
    if s in {"no", "n", "false"}:
        return False
    return None  # NA / nan / unknown -> not provided


def _date(v) -> Optional[date]:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v)[:10]).date()
    except ValueError:
        return None


def applicable_policies(sp: dict) -> list[str]:
    form = sp.get("form", {})
    ids = ["POL-01", "POL-02", "POL-03", "POL-04"]
    if _yes(form.get("external_speaker")):
        ids.append("POL-05")
    if (sp.get("expected_attendance") or 0) > 200:
        ids.append("POL-06")
    if sp.get("event_type") == "Charity Drive" or _yes(form.get("fundraising_registered")) is True:
        ids.append("POL-07")
    return ids


def policy_facts(sp: dict, retrieved_policy_ids: set[str]) -> list[dict]:
    form = sp.get("form", {})
    out: list[dict] = []

    def add(pid, rule, fact, ok):
        if pid in retrieved_policy_ids:
            out.append({"policy_id": pid, "rule": rule, "fact": fact, "satisfied": ok})

    budget = sp.get("budget_requested_inr")
    if budget is not None:
        if budget > 200000:
            ok = _yes(form.get("dean_approval"))
            add("POL-01", "Budgets above INR 200,000 need written Dean approval",
                f"Requested INR {budget:,.0f} (above ceiling); dean approval = {form.get('dean_approval')}", ok if ok is not None else None)
        else:
            add("POL-01", "Budgets above INR 200,000 need written Dean approval", f"Requested INR {budget:,.0f} is within the ceiling", True)
    else:
        add("POL-01", "Budget ceiling", "Requested budget not provided", None)

    lead = form.get("lead_time_days")
    if lead is None and _date(form.get("submission_date")) and _date(sp.get("event_date")):
        lead = (_date(sp["event_date"]) - _date(form["submission_date"])).days
    add("POL-02", "Submit at least 21 days before the event",
        f"Submitted {int(lead)} days before the event" if lead is not None else "Submission date / lead time not provided",
        None if lead is None else lead >= 21)

    ev = _date(sp.get("event_date"))
    if ev:
        md = (ev.month, ev.day)
        blackout = (11, 20) <= md <= (12, 15) or (4, 20) <= md <= (5, 10)
        add("POL-03", "No events 20 Nov-15 Dec or 20 Apr-10 May", f"Event date {ev.isoformat()} {'falls inside' if blackout else 'is outside'} an exam blackout window", not blackout)
    else:
        add("POL-03", "Exam blackout", "Event date not provided", None)

    att = sp.get("expected_attendance")
    cap = form.get("venue_capacity") or CAPACITY.get(sp.get("venue") or "")
    add("POL-04", "Expected attendance must not exceed venue capacity",
        f"Expected {att} vs capacity {int(cap)} at {sp.get('venue')}" if att and cap else "Attendance or venue capacity not provided",
        None if not (att and cap) else att <= cap)

    if _yes(form.get("external_speaker")):
        ap = _yes(form.get("speaker_approval"))
        add("POL-05", "External speaker/judge needs prior faculty-advisor approval",
            f"External speaker = Yes; faculty approval = {form.get('speaker_approval') or 'not provided'}", ap)

    if att and att > 200:
        sf = _yes(form.get("safety_plan"))
        add("POL-06", "Events above 200 attendees must attach a safety & security plan",
            f"Expected {att} attendees; safety plan = {form.get('safety_plan') or 'not provided'}", sf)

    fr = _yes(form.get("fundraising_registered"))
    if sp.get("event_type") == "Charity Drive" or fr is not None:
        add("POL-07", "Fundraising must be registered with the Finance Office",
            f"Fundraising registered = {form.get('fundraising_registered') or 'not provided'}", fr)
    return out
