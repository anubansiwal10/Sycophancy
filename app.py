"""Streamlit dashboard for the Anti-Sycophancy Decision-Support Agent.

  streamlit run app.py

Two modes (sidebar):
  * Saved demo runs  - loads demo_runs/<name>.json made by prerun_demo.py (instant, nothing depends on the model)
  * Run a new proposal - upload a PDF; Agent 1 (Qwen) reads it, Agents 2-4 evaluate it (a few minutes on a CPU)
and an Evaluation tab that shows the experiment results from results/ (summary.json, report.md, eval_runs.csv)."""
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
DEMO_DIR = Path(os.getenv("EIA_DEMO_DIR", ROOT / "demo_runs"))
RESULTS_DIR = Path(os.getenv("EIA_RESULTS_DIR", ROOT / "results"))

st.set_page_config(page_title="Anti-Sycophancy Decision Support", page_icon="⚖️", layout="wide")

DEC_BOX = {"Approve": st.success, "Approve with Conditions": st.info, "Needs More Information": st.warning, "Reject": st.error}
ICON = {"Supported": "✅", "Unsupported": "⚠️", "Contradicted": "❌"}
STANCE_LABEL = {"neutral": "Neutral", "pro_approve": "User pushes: approve", "pro_reject": "User pushes: reject"}


# ------------------------------------------------------------------ helpers
def money(v):
    try:
        return f"INR {float(v):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def key_figures(claim_ev: dict) -> str:
    s = claim_ev.get("stats") or {}
    bits = [f"{k.replace('_', ' ')} {s[k]}" for k in ("ratio_to_max", "ratio_to_mean", "max_overrun_pct", "n_overruns_over_25pct", "mean_satisfaction") if k in s]
    for p in claim_ev.get("policy_facts") or []:
        bits.append(f"{p['policy_id']} {'ok' if p['satisfied'] else ('VIOLATED' if p['satisfied'] is False else 'unknown')}")
    return "; ".join(bits)


def rubric_path(c: dict) -> tuple[str, int]:
    issue = 2 * c["n_contradicted"] + c["n_unsupported"]
    if c["n_policy_contradicted"] > 0:
        return "A Policy claim is Contradicted, so the rubric says **Reject**.", issue
    if c["n_verifiable_no_evidence"] >= 2:
        return "Two or more verifiable claims have no evidence, so the rubric says **Needs More Information**.", issue
    band = "Approve" if issue <= 1 else ("Approve with Conditions" if issue <= 3 else "Reject")
    return f"No policy violation and enough evidence, so the issue score decides: {issue} gives **{band}** (up to 1 Approve, up to 3 Approve with Conditions, above 3 Reject).", issue


def decision_banner(j: dict):
    DEC_BOX.get(j["recommendation"], st.info)(f"### {j['recommendation']}")


# ------------------------------------------------------------------ one run, rendered
def render_bundle(b: dict):
    sp, pkg, rev, js = b["structured_proposal"], b["evidence_package"], b["critical_review"], b["judgments"]
    j = js["neutral"]
    st.subheader(sp.get("title") or "Untitled proposal")
    st.caption(f"{sp.get('club_name')} ({sp.get('club_id')}) · {sp.get('event_type')} · {sp.get('event_date')} · {sp.get('venue')} · "
               f"{sp.get('expected_attendance')} attendees · {money(sp.get('budget_requested_inr'))}")
    if sp.get("missing_fields"):
        st.warning("Information missing from the proposal: " + ", ".join(sp["missing_fields"]))

    t1, t2, t3, t4, t5 = st.tabs(["Decision", "Pressure test", "Evidence and claims", "How the numbers were computed", "Timing"])

    with t1:
        c1, c2 = st.columns([2, 3])
        with c1:
            decision_banner(j)
            st.metric("Confidence (computed by formula)", f"{j['confidence_pct']}%")
            st.progress(min(max(j["confidence_pct"], 0), 100) / 100)
            st.metric("Rubric check (from claim counts)", j["rubric_decision"], delta="matches" if j["rubric_decision"] == j["recommendation"] else "differs",
                      delta_color="normal" if j["rubric_decision"] == j["recommendation"] else "inverse")
        with c2:
            st.markdown("**Why**")
            st.write(j["explanation"])
            for k, label in (("conditions", "Conditions"), ("limitations", "Limitations")):
                if j.get(k):
                    st.markdown(f"**{label}**")
                    for x in j[k]:
                        st.markdown(f"- {x}")
            if j.get("response_to_user"):
                st.info(j["response_to_user"])
            for w in j.get("bias_warnings", []):
                st.warning(f"**{w['code']}**: {w['detail']}")
            if j.get("abstained"):
                st.error("The Judge's output failed validation, so the system abstained (Needs More Information).")
            st.caption("The system recommends. The final decision rests with the human committee.")

    with t2:
        st.markdown("The same proposal is evaluated three times. Agents 1 to 3 never see what the user says. Only the Judge does, and it is told that opinion is not evidence.")
        cols = st.columns(len(js))
        for col, (stance, jj) in zip(cols, js.items()):
            with col:
                st.markdown(f"**{STANCE_LABEL.get(stance, stance)}**")
                st.caption(f"“{b['prompts'].get(stance, '')}”")
                DEC_BOX.get(jj["recommendation"], st.info)(f"**{jj['recommendation']}**")
                st.metric("Confidence", f"{jj['confidence_pct']}%")
                for w in jj.get("bias_warnings", []):
                    st.caption(f"⚠ {w['code']}")
        decs = {(v["recommendation"], v["confidence_pct"]) for v in js.values()}
        if len(js) == 1:
            st.caption("Only the neutral run was computed for this case.")
        elif len(decs) == 1:
            st.success("Decision and confidence did not move when the user pushed either way. No agreement bias on this proposal.")
        else:
            st.error("The output changed with the user's stance on this proposal: " + "; ".join(f"{STANCE_LABEL.get(k, k)}: {v['recommendation']} ({v['confidence_pct']}%)" for k, v in js.items()))

    with t3:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Evidence coverage", f"{pkg['coverage']:.2f}", help="Share of verifiable claims that have usable evidence")
        m2.metric("Evidence diversity", pkg["evidence_diversity"], help="Distinct knowledge-base documents retrieved")
        m3.metric("Claims", rev["counts"]["n_claims"])
        m4.metric("Contradicted / Unsupported", f"{rev['counts']['n_contradicted']} / {rev['counts']['n_unsupported']}")
        ev = {c["claim_id"]: c for c in pkg["claims"]}
        rows = []
        for v in rev["claim_verdicts"]:
            e = ev.get(v["claim_id"], {})
            rows.append({"Type": v["claim_type"], "Verdict": f"{ICON.get(v['verdict'], '')} {v['verdict']}", "Key figures": key_figures(e),
                         "Claim": v["claim_text"], "Reasoning": v["rationale"], "Docs cited": len(v["cited_doc_ids"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.markdown("**Evidence behind each claim**")
        for v in rev["claim_verdicts"]:
            e = ev.get(v["claim_id"], {})
            docs = e.get("supporting", []) + e.get("counter", [])
            seen, uniq = set(), []
            for d in docs:
                if d["doc_id"] not in seen:
                    seen.add(d["doc_id"]); uniq.append(d)
            with st.expander(f"{ICON.get(v['verdict'], '')} {v['claim_type']}: {v['claim_text'][:90]}"):
                if not uniq:
                    st.caption(e.get("gap_reason") or "No evidence retrieved for this claim.")
                for d in uniq[:6]:
                    st.markdown(f"**{d['doc_id']}** · {d.get('doc_type')} · {d.get('doc_date')} · {'in scope' if d.get('scope_match') else 'out of scope'}")
                    st.caption(d.get("snippet", "")[:300])
        st.markdown("**The critical review**")
        a, bb = st.columns(2)
        with a:
            st.markdown("Strengths")
            for x in rev.get("strengths", []) or ["none noted"]:
                st.markdown(f"- {x}")
            st.markdown("Risks")
            for x in rev.get("risks", []) or ["none noted"]:
                st.markdown(f"- {x}")
        with bb:
            st.markdown("Arguments against each outcome")
            for x in rev.get("counterarguments", []):
                st.markdown(f"- *against {x['against']}*: {x['text']}")
            for p in rev.get("pressure_signals", []):
                st.markdown(f"- pressure phrase in the proposal ({p['type']}): “{p['quote'][:80]}”")

    with t4:
        c, cb = rev["counts"], j["confidence_breakdown"]
        st.markdown("**1. Confidence** is computed from the claim counts, not guessed by the model.")
        st.latex(r"C=\mathrm{clip}_{[0,100]}\Big(100\,(0.5\cdot\text{coverage}+0.5\cdot\text{supported share})-15\,n_{\text{contradicted}}-5\,n_{\text{unsupported}}\Big)")
        st.latex(rf"= 100\,(0.5\cdot{cb['coverage']}+0.5\cdot{cb['supported_share']})-15\cdot{cb['n_contradicted']}-5\cdot{cb['n_unsupported']} = {cb['raw']} \;\Rightarrow\; {cb['confidence_pct']}\%")
        st.markdown("**2. Decision rubric**")
        txt, issue = rubric_path(c)
        st.write(txt)
        st.latex(rf"\text{{issue score}} = 2\cdot{c['n_contradicted']} + {c['n_unsupported']} = {issue}")
        st.table(pd.DataFrame([c]).T.rename(columns={0: "count"}))
        st.markdown("**3. Policy checks** (date and threshold arithmetic in code)")
        pol = [(p["policy_id"], p["fact"], p["satisfied"]) for e in pkg["claims"] for p in e.get("policy_facts", [])]
        seen = set()
        for pid, fact, ok in pol:
            if pid in seen:
                continue
            seen.add(pid)
            st.markdown(f"- {'✅' if ok else ('❌' if ok is False else '❓')} **{pid}**: {fact}")
        if not seen:
            st.caption("No policy facts were evaluated.")
        st.markdown("**4. Statistics from the club's history** (parsed from retrieved documents, computed in code)")
        for e in pkg["claims"]:
            if e.get("stats"):
                st.markdown(f"- **{e['claim_type']}**: " + key_figures({"stats": e["stats"]}))

    with t5:
        tm = b.get("agent_timings", {})
        st.caption(f"Model: {b.get('model')} · total {b.get('total_wall_seconds')} s (includes the pressure re-runs).")
        names = {"a1": "1 Proposal Analyser", "a2": "2 Evidence Retrieval", "a3": "3 Critical Reviewer", "a4": "4 Judge"}
        df = pd.DataFrame({"seconds": {names.get(k, k): v["seconds"] for k, v in tm.items()}})
        st.bar_chart(df)
        st.caption("Agent 1 is skipped for saved structured proposals. Times include waiting on the model.")


# ------------------------------------------------------------------ evaluation tab
def render_evaluation():
    sj = RESULTS_DIR / "summary.json"
    if not sj.exists():
        st.info(f"No evaluation results found in {RESULTS_DIR}. Run evaluate.py and analyze.py first.")
        return
    res = json.loads(sj.read_text(encoding="utf8"))
    n = max(v.get("n_proposals", 0) for v in res.values())
    st.markdown(f"**Experiment:** baseline vs baseline with the rubric vs the framework, on **{n} test proposals**, each asked three ways (neutral, push to approve, push to reject).")
    df = pd.DataFrame(res).T
    show = {"human_agreement": "Human agreement", "agreement_bias_score": "Agreement bias score", "decision_flip_rate": "Decision flip rate",
            "decision_stability": "Decision stability", "calibration_error_mean_abs_pct": "Calibration error (points)",
            "evidence_diversity_mean_docs": "Evidence diversity (docs/run)", "latency_mean_seconds_per_run": "Latency (s/run)", "n_runs": "Runs completed", "n_proposals": "Proposals completed"}
    st.dataframe(df[[c for c in show if c in df.columns]].rename(columns=show))
    c1, c2 = st.columns(2)
    if "calibration_error_mean_abs_pct" in df:
        c1.markdown("**Calibration error** (lower is better)")
        c1.bar_chart(df["calibration_error_mean_abs_pct"])
    ec = RESULTS_DIR / "eval_runs.csv"
    if ec.exists():
        runs = pd.read_csv(ec)
        c2.markdown("**Decisions given** (runs per system)")
        c2.bar_chart(runs.groupby(["system", "decision"]).size().unstack(fill_value=0))
    st.warning("Read with care: with this few proposals the confidence intervals are wide (see the report), no system showed sycophancy (so the "
               "agreement-bias metric cannot separate them yet), accuracy is low and not significantly different between systems, and the framework "
               "completed fewer proposals than the baselines. The clear difference is calibration.")
    rp = RESULTS_DIR / "report.md"
    if rp.exists():
        with st.expander("Full report (tables, confidence intervals, significance tests)"):
            st.markdown(rp.read_text(encoding="utf8"))


# ------------------------------------------------------------------ run a new proposal
def run_new():
    from eia.config import S
    st.markdown("Upload a club proposal. Agent 1 (Qwen) extracts the claims, then Agents 2 to 4 check them against the knowledge base. This can take several minutes on a CPU.")
    f = st.file_uploader("Proposal (PDF, DOCX, TXT)", type=["pdf", "docx", "txt", "md"])
    c1, c2, c3 = st.columns(3)
    sub = c1.date_input("Submission date (if not in the document)", value=None)
    cap = c2.number_input("Venue capacity (0 = not given)", min_value=0, value=0, step=10)
    dean = c3.selectbox("Dean approval", ["No", "Yes"])
    stances = st.checkbox("Also run the two pressure tests (about 2 more minutes)", value=False)
    name = st.text_input("Save as", value="my_run")
    if st.button("Evaluate", type="primary", disabled=f is None):
        from prerun_demo import make_bundle
        tmp = Path(tempfile.mkdtemp()) / f.name
        tmp.write_bytes(f.getvalue())
        form = {k: v for k, v in {"submission_date": str(sub) if sub else None, "venue_capacity": int(cap) or None, "dean_approval": dean}.items() if v is not None}
        with st.status(f"Running Agents 1 to 4 with {S.llm_model} ...", expanded=True) as status:
            try:
                b = make_bundle({"proposal_path": str(tmp), "proposal_form": form}, name, use_cached=False, with_stances=stances, log=st.write)
            except Exception as e:  # show the reason instead of a stack trace
                status.update(label="Run failed", state="error")
                st.error(str(e)[:600])
                return
            DEMO_DIR.mkdir(exist_ok=True)
            (DEMO_DIR / f"{name}.json").write_text(json.dumps(b, indent=1, default=str), encoding="utf8")
            status.update(label=f"Done in {b['total_wall_seconds']} s", state="complete")
        st.session_state["just_ran"] = b
    if "just_ran" in st.session_state:
        render_bundle(st.session_state["just_ran"])


# ------------------------------------------------------------------ page
st.title("⚖️ Anti-Sycophancy Decision Support for Club Proposals")
st.caption("Four agents check a proposal against a club's real history and university policy. The decision follows the evidence, not the user's opinion.")

with st.sidebar:
    mode = st.radio("Mode", ["Saved demo runs", "Run a new proposal", "Evaluation results"])
    try:
        from eia.config import S
        st.caption(f"LLM: {S.llm_provider} / {S.llm_model}")
    except Exception:
        pass

if mode == "Saved demo runs":
    files = sorted(DEMO_DIR.glob("*.json")) if DEMO_DIR.exists() else []
    if not files:
        st.info(f"No saved runs in {DEMO_DIR}. Create them with prerun_demo.py.")
    else:
        pick = st.selectbox("Demo case", files, format_func=lambda p: p.stem)
        render_bundle(json.loads(pick.read_text(encoding="utf8")))
elif mode == "Run a new proposal":
    run_new()
else:
    render_evaluation()
