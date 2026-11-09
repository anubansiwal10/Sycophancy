"""Excel dataset loader. Only the KnowledgeBase (evidence), Clubs, Proposals (text + submission form),
Config (scoring weights) and, for evaluation only, GroundTruth / PromptVariants are used.

Gold columns (scenario, sycophancy_trigger, Claims verdicts, EvidenceLinks) are NEVER read by the agents."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import pandas as pd

from .config import S

FORM_COLUMNS = ["submission_date", "lead_time_days", "venue", "venue_capacity", "expected_attendance",
                "budget_requested_inr", "budget_per_attendee_inr", "external_speaker", "speaker_approval",
                "safety_plan", "fundraising_registered", "dean_approval", "duration_hours"]


@dataclass
class Dataset:
    clubs: pd.DataFrame
    past_events: pd.DataFrame
    kb: pd.DataFrame
    proposals: pd.DataFrame
    ground_truth: pd.DataFrame
    prompt_variants: pd.DataFrame
    config: dict


@lru_cache(maxsize=1)
def load_dataset(path: Optional[str] = None) -> Dataset:
    sheets = pd.read_excel(path or S.dataset_path, sheet_name=None)
    cfg = {str(r.parameter): float(r.value) for r in sheets["Config"].itertuples()}
    return Dataset(
        clubs=sheets["Clubs"], past_events=sheets["PastEvents"], kb=sheets["KnowledgeBase"],
        proposals=sheets["Proposals"], ground_truth=sheets["GroundTruth"],
        prompt_variants=sheets["PromptVariants"], config=cfg,
    )


def event_types() -> list[str]:
    return sorted(load_dataset().past_events["event_type"].dropna().unique().tolist())


def resolve_club_id(club_name: Optional[str]) -> Optional[str]:
    """Map a free-text club name to a club_id in the dataset (case/spacing/typo tolerant)."""
    if not club_name:
        return None
    clubs = load_dataset().clubs
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
    names = {norm(n): cid for n, cid in zip(clubs["club_name"], clubs["club_id"])}
    key = norm(club_name)
    if key in names:
        return names[key]
    m = difflib.get_close_matches(key, list(names), n=1, cutoff=0.75)
    return names[m[0]] if m else None


def normalize_event_type(et: Optional[str]) -> Optional[str]:
    if not et:
        return None
    known = {e.lower(): e for e in event_types()}
    if et.strip().lower() in known:
        return known[et.strip().lower()]
    m = difflib.get_close_matches(et.strip().lower(), list(known), n=1, cutoff=0.7)
    return known[m[0]] if m else None


def benchmark_proposal(proposal_id: str) -> tuple[str, dict]:
    """(proposal_text, submission-form dict) for a benchmark proposal - no gold labels returned."""
    df = load_dataset().proposals
    row = df[df["proposal_id"] == proposal_id]
    if row.empty:
        raise KeyError(f"{proposal_id} not in Proposals sheet")
    r = row.iloc[0]
    form = {}
    for c in FORM_COLUMNS:
        v = r[c]
        form[c] = None if pd.isna(v) else (v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else (v.item() if hasattr(v, "item") else v))
    form["event_date"] = r["event_date"].strftime("%Y-%m-%d")
    form["title"] = r["title"]
    form["club_name"] = r["club_name"]
    form["event_type"] = r["event_type"]
    return str(r["proposal_text"]), form
