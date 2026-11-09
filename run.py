#!/usr/bin/env python
"""Run the four-agent pipeline on any club proposal.

  python run.py --proposal my_proposal.pdf                       # real Qwen analysis (demo mode)
  python run.py --proposal my_proposal.txt --form form.json      # optional submission-form fields
  python run.py --proposal-id PRP-0001 --cached                  # benchmark proposal, Agent-1 bypass (testing)
  python run.py --proposal-id PRP-0001 --user-prompt "I think it is excellent, please confirm my view."
"""
import argparse
import json
import sys
from pathlib import Path

from eia.config import S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", help="path to .txt/.md/.pdf/.docx/.json proposal")
    ap.add_argument("--proposal-id", help="benchmark proposal id from the Excel dataset (text + form fields, no gold labels)")
    ap.add_argument("--analysis-json", help="Agent-1 bypass for ANY proposal: path to a ready-made structured-proposal JSON (see data/cached_analysis/*.json)")
    ap.add_argument("--agent", type=int, choices=[1, 2, 3, 4], help="print the full output of just this agent after the run")
    ap.add_argument("--form", help="JSON file with submission-form fields (submission_date, venue_capacity, speaker_approval, ...)")
    ap.add_argument("--user-prompt", default="Evaluate this proposal and give a recommendation (Approve / Approve with Conditions / Reject / Needs More Information) with reasons.")
    ap.add_argument("--cached", action="store_true", help="Agent-1 bypass: load data/cached_analysis/<id>.json instead of calling Qwen")
    ap.add_argument("--llm", help="override EIA_LLM_PROVIDER (ollama|fake)")
    ap.add_argument("--model", help="override EIA_LLM_MODEL")
    ap.add_argument("--embed", help="override EIA_EMBED_PROVIDER (ollama|hash)")
    ap.add_argument("--json", action="store_true", help="print the full final state as JSON")
    a = ap.parse_args()
    if a.llm: S.llm_provider = a.llm
    if a.model: S.llm_model = a.model
    if a.embed: S.embed_provider = a.embed
    S.use_cached_analysis = a.cached or bool(a.analysis_json) or S.use_cached_analysis

    from eia.data import benchmark_proposal
    from eia.graph import agent_timings, run_pipeline

    inputs = {"user_prompt": a.user_prompt}
    if a.analysis_json:
        inputs["analysis_json_path"] = a.analysis_json
        inputs["proposal_text"] = ""
    elif a.proposal_id:
        text, form = benchmark_proposal(a.proposal_id)
        inputs.update(proposal_id=a.proposal_id, proposal_text=text, proposal_form=form)
    elif a.proposal:
        inputs["proposal_path"] = a.proposal
    else:
        sys.exit("give --proposal or --proposal-id")
    if a.form:
        inputs["proposal_form"] = {**inputs.get("proposal_form", {}), **json.loads(Path(a.form).read_text())}

    st = run_pipeline(inputs)
    S.runs_dir.mkdir(parents=True, exist_ok=True)
    out = S.runs_dir / f"run_{st['run_id']}.json"
    out.write_text(json.dumps({k: v for k, v in st.items() if k not in {"raw_hits", "ranked", "parsed_text"}}, indent=1, default=str))

    AGENT_KEYS = {1: ("proposal_analyser", "structured_proposal"), 2: ("evidence_retrieval", "evidence_package"),
                  3: ("critical_reviewer", "critical_review"), 4: ("judge", "judgment")}
    adir = S.runs_dir / f"run_{st['run_id']}_agents"
    adir.mkdir(parents=True, exist_ok=True)
    for n, (nm, key) in AGENT_KEYS.items():  # one file per agent = the individual output of each stage
        if st.get(key) is not None:
            (adir / f"agent{n}_{nm}.json").write_text(json.dumps(st[key], indent=1, default=str))
    if a.agent:
        print(json.dumps(st.get(AGENT_KEYS[a.agent][1]), indent=1, default=str))
        return
    if a.json:
        print(out.read_text())
        return
    j = st.get("judgment")
    if not j:
        print("PIPELINE STOPPED:", st.get("errors"), "| analysis feedback:", st.get("analysis_feedback"))
        return
    sp = st["structured_proposal"]
    print(f"\n=== {sp.get('title')} ({sp.get('club_name')}, {sp.get('club_id')}) | analysis: {st['analysis_source']} ===")
    if sp["missing_fields"]:
        print("Missing fields flagged for the user:", sp["missing_fields"])
    print("\nClaim verdicts:")
    for v in st["critical_review"]["claim_verdicts"]:
        print(f"  [{v['verdict']:<12}] {v['claim_id']} ({v['claim_type']}): {v['rationale'][:110]}")
    print(f"\nRECOMMENDATION: {j['recommendation']}   |   confidence (rule-based): {j['confidence_pct']}%   |   rubric check: {j['rubric_decision']}")
    print("\n" + j["explanation"])
    if j["response_to_user"]:
        print("\nTo the committee member:", j["response_to_user"])
    for w in j["bias_warnings"]:
        print(f"  ! {w['code']}: {w['detail']}")
    print("\nTiming:", json.dumps(agent_timings(st["trace"])), f"| total {st['total_seconds']}s")
    print("Per-agent outputs in", adir, "\nFull run saved to", out, "\nFinal decision rests with the human committee.")


if __name__ == "__main__":
    main()
