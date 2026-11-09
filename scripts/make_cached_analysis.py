#!/usr/bin/env python
"""Build data/cached_analysis/<proposal_id>.json (the Agent-1 bypass files) for the benchmark proposals.

Source = Proposals sheet (fields + submission form) and Claims sheet columns claim_id / claim_type / claim_text ONLY.
Verdicts, rationales, evidence_available, gold_evidence_doc_ids are deliberately NOT copied. Note for the paper: the
claim *segmentation and typing* therefore comes from the dataset annotation rather than from Qwen in this mode.
Replace these files with your own Agent-1 JSON dumps if you have them (key aliases are tolerated)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from eia.config import S
from eia.data import FORM_COLUMNS

sheets = pd.read_excel(S.dataset_path, sheet_name=["Proposals", "Claims"])
props, claims = sheets["Proposals"], sheets["Claims"][["claim_id", "proposal_id", "claim_type", "claim_text"]]
S.cached_dir.mkdir(parents=True, exist_ok=True)
by_prop = {k: g for k, g in claims.groupby("proposal_id")}
for r in props.itertuples(index=False):
    d = r._asdict()
    form = {}
    for c in FORM_COLUMNS:
        v = d[c]
        form[c] = None if pd.isna(v) else (v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else (v.item() if hasattr(v, "item") else v))
    split = lambda s: [x.strip() for x in str(s).split(";") if x.strip()]
    out = {"proposal_id": d["proposal_id"], "club_name": d["club_name"], "club_id": d["club_id"], "title": d["title"],
           "event_type": d["event_type"], "event_date": d["event_date"].strftime("%Y-%m-%d"), "venue": d["venue"],
           "expected_attendance": int(d["expected_attendance"]), "budget_requested_inr": float(d["budget_requested_inr"]),
           "objectives": split(d["objectives"]), "resources": split(d["resources"]), "risks": split(d["risks"]),
           "form": {k: v for k, v in form.items() if v is not None},
           "claims": [{"claim_id": c.claim_id, "claim_type": c.claim_type, "claim_text": c.claim_text} for c in by_prop[d["proposal_id"]].itertuples()]}
    (S.cached_dir / f"{d['proposal_id']}.json").write_text(json.dumps(out, indent=1))
print(f"wrote {len(props)} files to {S.cached_dir}")
