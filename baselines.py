#!/usr/bin/env python
"""Baseline answers for the dashboard: what a plain Qwen call and a Qwen call with the committee rubric say about the same
proposal, for the same three user messages that the Pressure test uses. The prompts are the ones the experiment used
(evaluate.baseline_run), so what the dashboard shows is what the experiment measured.

  python baselines.py demo_runs/badminton.json                 # add the baseline answers to one saved run
  python baselines.py --all                                    # every saved run that does not have them yet
  python baselines.py --all --force                            # recompute them

Writes a "baselines" key into the saved run (demo_runs/<name>.json). Nothing else in the file is touched. Needs the local
model (Ollama) running, like prerun_demo.py: 2 systems x 3 stances = 6 model calls, about 3 to 6 minutes on a CPU."""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

SYSTEMS = {"baseline": "Baseline", "baseline_rubric": "Baseline + rubric"}
STANCES = ["neutral", "pro_approve", "pro_reject"]
STANCE_LABEL = {"neutral": "Neutral", "pro_approve": "User pushes: approve", "pro_reject": "User pushes: reject"}
ROOT = Path(__file__).parent


def proposal_text_for(sp: dict) -> str:
    """The text the baseline reads. A proposal that came from a PDF/DOCX/TXT has its own text. A demo case that was loaded
    from structured JSON has none, so the same fields and claims that Agent 1 would have produced are written out as text."""
    text = (sp.get("proposal_text") or "").strip()
    if text:
        return text
    lines = [f"Proposal: {sp.get('title')}", f"Club: {sp.get('club_name')}", f"Event type: {sp.get('event_type')}",
             f"Event date: {sp.get('event_date')}", f"Venue: {sp.get('venue')}", f"Expected attendance: {sp.get('expected_attendance')}",
             f"Budget requested: INR {sp.get('budget_requested_inr')}"]
    for key, label in (("objectives", "Objectives"), ("resources", "Resources"), ("risks", "Risks")):
        if sp.get(key):
            lines.append(f"{label}: " + "; ".join(str(x) for x in sp[key]))
    claims = [c.get("claim_text") for c in sp.get("claims") or [] if c.get("claim_text")]
    if claims:
        lines.append("Statements made in the proposal:")
        lines += [f"- {c}" for c in claims]
    return "\n".join(lines)


def compute_baselines(sp: dict, prompts: dict | None = None, log=print) -> dict:
    """{'baseline': {stance: answer}, 'baseline_rubric': {stance: answer}} for one structured proposal. One model call each."""
    from evaluate import ORDER, baseline_run
    from prerun_demo import PROMPTS
    prompts = prompts or PROMPTS
    text, form = proposal_text_for(sp), sp.get("form") or {}
    out = {s: {} for s in SYSTEMS}
    log("  baselines: starting")
    for system in SYSTEMS:
        for stance in STANCES:
            t = time.perf_counter()
            try:
                r = baseline_run(text, form, prompts[stance], with_rubric=(system == "baseline_rubric"))
                dec = r.get("decision") if r.get("decision") in ORDER else "Needs More Information"  # same rule as evaluate.py
                try:
                    conf = float(r.get("confidence_pct") or 0)
                except (TypeError, ValueError):
                    conf = 0.0
                out[system][stance] = {"decision": dec, "confidence_pct": conf, "reasons": str(r.get("reasons") or ""),
                                       "factual_assertions": [str(x) for x in (r.get("factual_assertions") or [])],
                                       "seconds": round(time.perf_counter() - t, 1)}
                log(f"  baselines: {system} [{stance}] -> {dec} ({conf:g}%) in {out[system][stance]['seconds']}s")
            except Exception as e:  # keep going: a failed call is shown as failed, not hidden
                out[system][stance] = {"error": str(e)[:300], "seconds": round(time.perf_counter() - t, 1)}
                log(f"  baselines: {system} [{stance}] FAILED: {str(e)[:120]}")
    return out


def system_rows(b: dict) -> dict:
    """{system label: {stance: {'decision', 'confidence_pct', ...} or None}} for Baseline, Baseline + rubric and Spectrum."""
    bl = b.get("baselines") or {}
    rows = {label: {s: (bl.get(key) or {}).get(s) for s in STANCES} for key, label in SYSTEMS.items()}
    rows["Spectrum"] = {s: ({"decision": j["recommendation"], "confidence_pct": j["confidence_pct"]} if (j := (b.get("judgments") or {}).get(s)) else None)
                        for s in STANCES}
    return rows


def decisions_moved(row: dict) -> tuple[bool, int, int]:
    """(did the decision change with the user's push, how many of the two pushes moved it, how many pushes could be compared)."""
    neutral = (row.get("neutral") or {}).get("decision")
    pushes = [(row.get(s) or {}).get("decision") for s in ("pro_approve", "pro_reject")]
    compared = [p for p in pushes if p and neutral]
    moved = sum(p != neutral for p in compared)
    return moved > 0, moved, len(compared)


def write_json_atomic(path: Path, data: dict):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf8") as f:
        json.dump(data, f, indent=1, default=str)
    os.replace(tmp, path)


def ollama_problem() -> str | None:
    from eia.config import S
    if S.llm_provider != "ollama":
        return None
    import requests
    try:
        names = [m["name"] for m in requests.get(S.ollama_base_url + "/api/tags", timeout=4).json().get("models", [])]
    except Exception:
        return f"Ollama is not reachable at {S.ollama_base_url}. Start the Ollama app, then try again."
    if S.llm_model not in names and f"{S.llm_model}:latest" not in names:
        return f"The model '{S.llm_model}' is not installed. Run: ollama pull {S.llm_model}"
    return None


def main():
    os.environ.setdefault("EIA_PROMPT_DOCS_PER_CLAIM", "2")
    os.environ.setdefault("EIA_SNIPPET_CHARS", "120")
    os.environ.setdefault("EIA_LLM_TIMEOUT", "1800")
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="saved runs, e.g. demo_runs/badminton.json")
    ap.add_argument("--all", action="store_true", help="every JSON in --dir")
    ap.add_argument("--dir", default=str(ROOT / "demo_runs"))
    ap.add_argument("--force", action="store_true", help="recompute even if the run already has baseline answers")
    a = ap.parse_args()
    files = [Path(f) for f in a.files] + (sorted(Path(a.dir).glob("*.json")) if a.all else [])
    if not files:
        raise SystemExit("give a saved run (demo_runs/<name>.json) or --all")
    problem = ollama_problem()
    if problem:
        raise SystemExit(problem)
    for p in files:
        d = json.loads(p.read_text(encoding="utf8"))
        if d.get("baselines") and not a.force:
            print(f"{p.name}: already has baseline answers (use --force to recompute)")
            continue
        print(f"{p.name}: running baseline and baseline + rubric ...", flush=True)
        t = time.perf_counter()
        d["baselines"] = compute_baselines(d["structured_proposal"], d.get("prompts"), log=lambda m: print(m, flush=True))
        d["baseline_seconds"] = round(time.perf_counter() - t, 1)
        write_json_atomic(p, d)
        print(f"{p.name}: saved ({d['baseline_seconds']} s)")


if __name__ == "__main__":
    main()
