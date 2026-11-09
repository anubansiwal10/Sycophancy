"""Typed contracts passed between the four agents."""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

CLAIM_TYPES = ["Attendance", "Budget", "Past Performance", "Policy", "Impact"]
VERIFIABLE_TYPES = {"Attendance", "Budget", "Past Performance", "Policy"}
DECISIONS = ["Approve", "Approve with Conditions", "Reject", "Needs More Information"]
VERDICTS = ["Supported", "Unsupported", "Contradicted"]
YES_NO_FIELDS = ["external_speaker", "speaker_approval", "safety_plan", "fundraising_registered", "dean_approval"]


def _num(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.sub(r"[^\d.\-]", "", str(v))
    try:
        return float(m) if m else None
    except ValueError:
        return None


# ---------------------------------------------------------------- Agent 1 -------------------
class ExtractedClaim(BaseModel):
    claim_type: str = "Impact"
    claim_text: str

    @field_validator("claim_type")
    @classmethod
    def _t(cls, v: str) -> str:
        for t in CLAIM_TYPES:
            if v.strip().lower() == t.lower():
                return t
        return "Impact"  # anything the model cannot classify is treated as non-verifiable


class ExtractedProposal(BaseModel):
    """What Qwen must return. Everything is nullable: the model must not invent values."""
    club_name: Optional[str] = None
    title: Optional[str] = None
    event_type: Optional[str] = None
    event_date: Optional[str] = None
    submission_date: Optional[str] = None
    venue: Optional[str] = None
    expected_attendance: Optional[int] = None
    budget_requested_inr: Optional[float] = None
    external_speaker: Optional[str] = None
    speaker_approval: Optional[str] = None
    safety_plan: Optional[str] = None
    fundraising_registered: Optional[str] = None
    dean_approval: Optional[str] = None
    duration_hours: Optional[float] = None
    objectives: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)

    @field_validator("expected_attendance", mode="before")
    @classmethod
    def _att(cls, v):
        n = _num(v)
        return int(n) if n is not None else None

    @field_validator("budget_requested_inr", "duration_hours", mode="before")
    @classmethod
    def _f(cls, v):
        return _num(v)

    @field_validator("objectives", "resources", "risks", mode="before")
    @classmethod
    def _l(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [s.strip() for s in re.split(r";|\n", v) if s.strip()]
        return [str(x) for x in v]


class Claim(BaseModel):
    claim_id: str
    claim_type: str
    claim_text: str
    verifiable: bool = True
    decision_criteria: str = ""
    evidence_needed: list[str] = Field(default_factory=list)
    source_ref: str = ""


class StructuredProposal(BaseModel):
    """Agent-1 output = the 'Structured Proposal' that Agents 2-4 consume (also the cached-JSON format)."""
    proposal_id: str = "PRP-NEW"
    club_name: Optional[str] = None
    club_id: Optional[str] = None
    title: Optional[str] = None
    event_type: Optional[str] = None
    event_date: Optional[str] = None
    venue: Optional[str] = None
    expected_attendance: Optional[int] = None
    budget_requested_inr: Optional[float] = None
    budget_per_attendee_inr: Optional[float] = None
    objectives: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    form: dict[str, Any] = Field(default_factory=dict)  # submission_date, lead_time_days, venue_capacity, speaker/safety/fundraising/dean flags
    missing_fields: list[str] = Field(default_factory=list)
    proposal_text: str = ""


# ---------------------------------------------------------------- Agent 3 -------------------
class ClaimVerdict(BaseModel):
    claim_id: str
    verdict: str
    rationale: str = ""
    cited_doc_ids: list[str] = Field(default_factory=list)

    @field_validator("verdict")
    @classmethod
    def _v(cls, v: str) -> str:
        for t in VERDICTS:
            if v.strip().lower() == t.lower():
                return t
        raise ValueError(f"verdict must be one of {VERDICTS}")


class ReviewVerdicts(BaseModel):
    verdicts: list[ClaimVerdict]


class Counterargument(BaseModel):
    against: str = "proposal"  # proposal | rejection
    text: str


class PressureSignal(BaseModel):
    type: str = "other"
    quote: str = ""


class Challenge(BaseModel):
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    counterarguments: list[Counterargument] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    pressure_signals: list[PressureSignal] = Field(default_factory=list)


# ---------------------------------------------------------------- Agent 4 -------------------
class Assertion(BaseModel):
    statement: str
    claim_id: Optional[str] = None
    kind: str = "evidence_backed"  # evidence_backed | gap_acknowledgement
    doc_ids: list[str] = Field(default_factory=list)


class JudgeDraft(BaseModel):
    recommendation: str
    explanation: str
    conditions: list[str] = Field(default_factory=list)
    assertions: list[Assertion] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    response_to_user: str = ""

    @field_validator("recommendation")
    @classmethod
    def _r(cls, v: str) -> str:
        for d in DECISIONS:
            if v.strip().lower() == d.lower():
                return d
        raise ValueError(f"recommendation must be one of {DECISIONS}")
