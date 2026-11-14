#!/usr/bin/env python
"""Print the numbers behind a run: coverage, evidence, claim verdicts, confidence formula, rubric check, warnings.

  python show_run.py              # newest run in ./runs
  python show_run.py runs\run_ab12cd34.json
"""
import json
import sys
from pathlib import Path


def newest(runs: Path) -> Path:
    files = sorted(runs.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"no run_*.json files in {runs}")
    return files[-1]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest(Path("runs"))
    st = json.loads(path.read_text(encoding="utf8"))
    sp, pkg, rev, j = st.get("structured_proposal") or {}, st.get("evidence_package") or {}, st.get("critical_review") or {}, st.get("judgment") or {}
    line = "-" * 78
    print(f"\n{line}\nRUN {path.name} | {sp.get('title')} | {sp.get('club_name')} ({sp.get('club_id')}) | {sp.get('event_type')}\n{line}")
    print(f"Time: {st.get('total_seconds')} s    stance seen by Judge: {j.get('user_stance')}")

    print(f"\nAGENT 2 - EVIDENCE\n  coverage            = {pkg.get('coverage')}   "
          f"({pkg.get('n_verifiable', 0) - pkg.get('n_verifiable_no_evidence', 0)} of {pkg.get('n_verifiable')} verifiable claims have usable evidence)")
    print(f"  evidence diversity  = {pkg.get('evidence_diversity')} distinct documents   (retrieval attempts: {pkg.get('attempts')})")
    for c in pkg.get("claims", []):
        s = c.get("stats") or {}
        keys = [k for k in ("ratio_to_max", "ratio_to_mean", "max_overrun_pct", "n_overruns_over_25pct", "mean_satisfaction") if k in s]
        facts = ", ".join(f"{k}={s[k]}" for k in keys)
        pf = "; ".join(f"{p['policy_id']}={'ok' if p['satisfied'] else ('VIOLATED' if p['satisfied'] is False else 'unknown')}" for p in c.get("policy_facts", []))
        print(f"  - {c['claim_id']} [{c['claim_type']}] evidence_available={c['evidence_available']} {facts}{pf}")

    print("\nAGENT 3 - CLAIM VERDICTS")
    for v in rev.get("claim_verdicts", []):
        print(f"  - {v['claim_id']} [{v['claim_type']}] {v['verdict']}  cites {len(v['cited_doc_ids'])} docs")
    print(f"  counts: {json.dumps(rev.get('counts'))}")
    for fl in rev.get("flags", []):
        print(f"  flag: {fl}")
    for p in rev.get("pressure_signals", []):
        print(f"  pressure signal in proposal: {p['type']}: \"{p['quote'][:70]}\"")

    cb = j.get("confidence_breakdown") or {}
    print("\nAGENT 4 - JUDGE")
    print(f"  recommendation      = {j.get('recommendation')}      rubric_decision (from the counts) = {j.get('rubric_decision')}")
    print(f"  confidence          = {j.get('confidence_pct')}%   raw {cb.get('raw')}  "
          f"= 100*(0.5*{cb.get('coverage')} + 0.5*{cb.get('supported_share')}) - 15*{cb.get('n_contradicted')} - 5*{cb.get('n_unsupported')}")
    c = (rev.get("counts") or {})
    if c:
        print(f"  issue score         = 2*{c.get('n_contradicted')} + {c.get('n_unsupported')} = {2 * c.get('n_contradicted', 0) + c.get('n_unsupported', 0)}")
    print(f"  abstained (failed validation) = {j.get('abstained')}   judge attempts = {j.get('judge_attempts')}")
    print(f"  assertions: {j.get('claims_asserted')}, of which citing no document: {j.get('unsupported_claims_asserted')}")
    print(f"  bias warnings: {[w['code'] for w in j.get('bias_warnings', [])] or 'none'}")
    for w in j.get("bias_warnings", []):
        print(f"     {w['code']}: {w['detail']}")
    print(f"  final decision authority: {j.get('final_decision_authority')}")
    print("\n  per-node timing (s):", {t["node"]: t["seconds"] for t in st.get("trace", [])})


if __name__ == "__main__":
    main()
