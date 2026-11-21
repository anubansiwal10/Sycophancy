"""Spectrum: dashboard for the Anti-Sycophancy Decision-Support Agent.

  python -m streamlit run app.py

Three pages (top navigation):
  * Saved runs          - loads demo_runs/<name>.json made by prerun_demo.py (instant, nothing depends on the model).
                          The "Baselines vs Spectrum" tab shows what a plain Qwen call and a Qwen call with the rubric answered
                          (added by prerun_demo.py --baselines, python baselines.py, or the button in that tab)
  * Evaluate a proposal - upload a PDF; Agent 1 (local Qwen) reads it, Agents 2-4 evaluate it. Runs in the background with live progress.
  * Results             - the experiment results from results_local/ (summary.json, report.md, eval_runs.csv)

Environment: EIA_DEMO_DIR (default demo_runs), EIA_RESULTS_DIR (default results_local). The model settings below default to the ones the
experiment used; set the environment variables yourself to override them."""
import hashlib
import html
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

# Settings used for the experiment. They must be set BEFORE anything from eia/ is imported (eia.config reads them once).
# Without them the pipeline falls back to larger prompts and a 15 minute timeout and a CPU run takes about 30 minutes.
os.environ.setdefault("EIA_PROMPT_DOCS_PER_CLAIM", "2")
os.environ.setdefault("EIA_SNIPPET_CHARS", "120")
os.environ.setdefault("EIA_LLM_TIMEOUT", "1800")

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
DEMO_DIR = Path(os.getenv("EIA_DEMO_DIR", ROOT / "demo_runs"))
RESULTS_DIR = Path(os.getenv("EIA_RESULTS_DIR", ROOT / "results_local"))   # the full 30-proposal experiment ("results" is an older 5-proposal test run)

st.set_page_config(page_title="Spectrum", layout="wide", initial_sidebar_state="collapsed")

PAGES = ["Saved runs", "Evaluate a proposal", "Results"]
STANCE_LABEL = {"neutral": "Neutral", "pro_approve": "User pushes: approve", "pro_reject": "User pushes: reject"}
DEC_KIND = {"Approve": "good", "Approve with Conditions": "info", "Needs More Information": "warn", "Reject": "bad"}
VERDICT_KIND = {"Supported": "good", "Unsupported": "warn", "Contradicted": "bad"}

CSS = """
<style>
html, body, [class*="st-"], button, input, textarea { font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif; }
/* The rule above also matched Streamlit's icon elements and replaced their icon font, so dropdown arrows showed as the words "arrow_drop_down". Give the icons their font back. */
[data-testid="stIconMaterial"], [data-testid*="Icon"], [translate="no"][aria-hidden="true"] { font-family: "Material Symbols Rounded", "Material Icons" !important; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { max-width: 1180px; padding-top: 1.4rem; padding-bottom: 4rem; }
.sp-brand { font-size: 1.75rem; font-weight: 750; letter-spacing: 0.01em; line-height: 1.1; margin: 0; }
.sp-brand span { display: block; font-size: 0.72rem; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.6; margin-top: 0.15rem; }
.sp-rule { border: 0; border-top: 1px solid rgba(128,128,128,0.28); margin: 0.6rem 0 1.4rem 0; }
.sp-page-title { font-size: 1.5rem; font-weight: 650; margin: 0 0 0.2rem 0; }
.sp-page-sub { opacity: 0.7; margin: 0 0 1.2rem 0; }
.sp-decision { border-left: 5px solid var(--c); background: color-mix(in srgb, var(--c) 11%, transparent); border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.9rem; }
.sp-decision .l { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.65; }
.sp-decision .v { font-size: 1.55rem; font-weight: 700; line-height: 1.25; }
.sp-decision.small .v { font-size: 1.15rem; }
.sp-good { --c: #1b9e6b; } .sp-info { --c: #3b6fd4; } .sp-warn { --c: #d28a0c; } .sp-bad { --c: #cf3b3b; }
.sp-badge { display: inline-block; padding: 0.08rem 0.55rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; color: var(--c); background: color-mix(in srgb, var(--c) 14%, transparent); border: 1px solid color-mix(in srgb, var(--c) 40%, transparent); }
.sp-step { display: flex; align-items: baseline; gap: 0.8rem; padding: 0.55rem 0; border-bottom: 1px solid rgba(128,128,128,0.2); }
.sp-step .n { font-weight: 650; min-width: 11.5rem; }
.sp-step .d { flex: 1; opacity: 0.75; }
.sp-step .s { min-width: 8.5rem; text-align: right; font-variant-numeric: tabular-nums; }
.sp-step.wait { opacity: 0.45; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
div[data-baseweb="tab-list"] { gap: 0.4rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ small helpers
def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def badge(text: str, kind: str) -> str:
    return f'<span class="sp-badge sp-{kind}">{esc(text)}</span>'


def decision_card(label: str, value: str, small: bool = False):
    st.markdown(f'<div class="sp-decision sp-{DEC_KIND.get(value, "info")}{" small" if small else ""}"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div></div>',
                unsafe_allow_html=True)


def page_header(title: str, sub: str):
    st.markdown(f'<div class="sp-page-title">{esc(title)}</div><div class="sp-page-sub">{esc(sub)}</div>', unsafe_allow_html=True)


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


# ------------------------------------------------------------------ baselines next to Spectrum
def render_baselines(b: dict, path: Path | None):
    from baselines import STANCES, compute_baselines, decisions_moved, proposal_text_for, system_rows, write_json_atomic
    js, sp = b["judgments"], b["structured_proposal"]
    st.markdown("The same proposal and the same three user messages as the Pressure test, given to a plain Qwen call (**Baseline**), "
                "to the same call with the committee rubric in its prompt (**Baseline + rubric**), and to **Spectrum**.")
    if not b.get("baselines"):
        st.info("The baseline answers have not been computed for this saved run.")
        st.caption("Ahead of a demo, run  python baselines.py --all  in a terminal. Or run them here: 6 short model calls, about 3 to 6 minutes on a CPU, then they are saved with the run.")
        if st.button("Run the baselines now", type="primary", key="run_baselines"):
            problem = preflight()
            if problem:
                st.error(problem)
                return
            t0 = time.time()
            with st.spinner("Running Baseline and Baseline + rubric on the three user messages (6 model calls)..."):
                b["baselines"] = compute_baselines(sp, b.get("prompts"), log=lambda m: None)
            b["baseline_seconds"] = round(time.time() - t0, 1)
            if path is not None:
                write_json_atomic(Path(path), b)
            st.rerun()
        return

    rows = system_rows(b)
    ref = (js.get("neutral") or {}).get("rubric_decision")
    if ref:
        st.caption(f"What the rubric gives from the evidence for this proposal: **{ref}**. Spectrum's decision is checked against it.")

    def cell(a):
        if not a:
            return "not run"
        if a.get("error"):
            return "failed"
        return f"{a['decision']} ({a['confidence_pct']:g}%)"

    table = pd.DataFrame([{"System": label, **{STANCE_LABEL[s]: cell(r[s]) for s in STANCES}} for label, r in rows.items()])
    st.dataframe(table, hide_index=True, width="stretch")

    cols = st.columns(len(rows), gap="large")
    for col, (label, r) in zip(cols, rows.items()):
        moved, n_moved, n_cmp = decisions_moved(r)
        with col:
            st.markdown(f"**{label}**")
            if n_cmp == 0:
                st.caption("Nothing to compare for this system.")
            elif moved:
                st.error(f"The decision moved with the user's push in {n_moved} of {n_cmp} cases.")
            else:
                st.success("The decision did not move when the user pushed either way.")

    st.markdown("**What each system answered**")
    for stance in STANCES:
        with st.expander(STANCE_LABEL[stance], expanded=(stance == "pro_approve")):
            st.caption(f"“{(b.get('prompts') or {}).get(stance, '')}”")
            cols = st.columns(len(rows), gap="large")
            for col, (label, r) in zip(cols, rows.items()):
                a = r.get(stance)
                with col:
                    st.markdown(f"**{label}**")
                    if not a:
                        st.caption("Not run for this case.")
                        continue
                    if a.get("error"):
                        st.error("The model call failed: " + a["error"])
                        continue
                    decision_card("Decision", a["decision"], small=True)
                    st.metric("Confidence reported", f"{a['confidence_pct']:g}%")
                    if label == "Spectrum":
                        jj = js[stance]
                        st.write(jj["explanation"])
                        if jj.get("response_to_user"):
                            st.info(jj["response_to_user"])
                    else:
                        if a.get("reasons"):
                            st.write(a["reasons"])
                        for x in a.get("factual_assertions") or []:
                            st.markdown(f"- {x}")
    note = "Confidence for Baseline and Baseline + rubric is whatever number the model wrote. Spectrum's is computed by formula from the evidence."
    if not (sp.get("proposal_text") or "").strip():
        note += " This case was loaded from structured data, so the two baselines read the proposal written out from the same fields and claims."
    st.caption(note)


# ------------------------------------------------------------------ one run, rendered
def render_bundle(b: dict, path: Path | None = None):
    sp, pkg, rev, js = b["structured_proposal"], b["evidence_package"], b["critical_review"], b["judgments"]
    j = js["neutral"]
    st.markdown(f"### {esc(sp.get('title') or 'Untitled proposal')}")
    st.caption(f"{sp.get('club_name')} ({sp.get('club_id')})  |  {sp.get('event_type')}  |  {sp.get('event_date')}  |  {sp.get('venue')}  |  "
               f"{sp.get('expected_attendance')} attendees  |  {money(sp.get('budget_requested_inr'))}")
    if sp.get("missing_fields"):
        st.warning("Information missing from the proposal: " + ", ".join(sp["missing_fields"]))

    t1, t2, tb, t3, t4, t5 = st.tabs(["Decision", "Pressure test", "Baselines vs Spectrum", "Evidence and claims", "How the numbers were computed", "Timing"])

    with t1:
        c1, c2 = st.columns([2, 3], gap="large")
        with c1:
            decision_card("Recommendation", j["recommendation"])
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
        cols = st.columns(len(js), gap="large")
        for col, (stance, jj) in zip(cols, js.items()):
            with col:
                st.markdown(f"**{STANCE_LABEL.get(stance, stance)}**")
                st.caption(f"“{b['prompts'].get(stance, '')}”")
                decision_card("Recommendation", jj["recommendation"], small=True)
                st.metric("Confidence", f"{jj['confidence_pct']}%")
                for w in jj.get("bias_warnings", []):
                    st.caption(f"Warning: {w['code']}")
        decs = {(v["recommendation"], v["confidence_pct"]) for v in js.values()}
        if len(js) == 1:
            st.caption("Only the neutral run was computed for this case.")
        elif len(decs) == 1:
            st.success("Decision and confidence did not move when the user pushed either way. No agreement bias on this proposal.")
        else:
            st.error("The output changed with the user's stance on this proposal: " + "; ".join(f"{STANCE_LABEL.get(k, k)}: {v['recommendation']} ({v['confidence_pct']}%)" for k, v in js.items()))

    with tb:
        render_baselines(b, path)

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
            rows.append({"Type": v["claim_type"], "Verdict": v["verdict"], "Key figures": key_figures(e),
                         "Claim": v["claim_text"], "Reasoning": v["rationale"], "Docs cited": len(v["cited_doc_ids"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown("**Evidence behind each claim**")
        for v in rev["claim_verdicts"]:
            e = ev.get(v["claim_id"], {})
            docs = e.get("supporting", []) + e.get("counter", [])
            seen, uniq = set(), []
            for d in docs:
                if d["doc_id"] not in seen:
                    seen.add(d["doc_id"]); uniq.append(d)
            with st.expander(f"[{v['verdict']}] {v['claim_type']}: {v['claim_text'][:90]}"):
                if not uniq:
                    st.caption(e.get("gap_reason") or "No evidence retrieved for this claim.")
                for d in uniq[:6]:
                    st.markdown(f"**{d['doc_id']}**  |  {d.get('doc_type')}  |  {d.get('doc_date')}  |  {'in scope' if d.get('scope_match') else 'out of scope'}")
                    st.caption(d.get("snippet", "")[:300])
        st.markdown("**The critical review**")
        a, bb = st.columns(2, gap="large")
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
        seen = set()
        for e in pkg["claims"]:
            for p in e.get("policy_facts", []):
                if p["policy_id"] in seen:
                    continue
                seen.add(p["policy_id"])
                ok = p["satisfied"]
                st.markdown(f"- {badge('Satisfied', 'good') if ok else (badge('Violated', 'bad') if ok is False else badge('Unknown', 'warn'))} **{esc(p['policy_id'])}**: {esc(p['fact'])}",
                            unsafe_allow_html=True)
        if not seen:
            st.caption("No policy facts were evaluated.")
        st.markdown("**4. Statistics from the club's history** (parsed from retrieved documents, computed in code)")
        for e in pkg["claims"]:
            if e.get("stats"):
                st.markdown(f"- **{e['claim_type']}**: " + key_figures({"stats": e["stats"]}))

    with t5:
        tm = b.get("agent_timings", {})
        st.caption(f"Model: {b.get('model')}  |  total {b.get('total_wall_seconds')} s (includes the pressure re-runs)." + (f" The baselines took {b['baseline_seconds']} s more." if b.get("baseline_seconds") else ""))
        names = {"a1": "1 Proposal Analyser", "a2": "2 Evidence Retrieval", "a3": "3 Critical Reviewer", "a4": "4 Judge"}
        df = pd.DataFrame({"seconds": {names.get(k, k): v["seconds"] for k, v in tm.items()}})
        st.bar_chart(df)
        st.caption("Agent 1 is skipped for saved structured proposals. Times include waiting on the model.")


# ------------------------------------------------------------------ background job for "Evaluate a proposal"
STAGES = [  # (agent key, name, what it does, typical seconds on the 16 GB CPU laptop)
    ("a1", "Agent 1  Proposal Analyser", "reads the PDF and extracts the claims", 65),
    ("a2", "Agent 2  Evidence Retrieval", "searches the knowledge base", 5),
    ("a3", "Agent 3  Critical Reviewer", "checks each claim against the evidence", 155),
    ("a4", "Agent 4  Judge", "writes the recommendation", 55),
]
LAST_NODE = {"a1_load_cached": 1, "a1_requirements_generator": 1, "a2_evidence_packager": 2, "a3_review_formatter": 3, "a4_bias_warning": 4}


class Job:
    """One pipeline run in a background thread. Survives page switches and reruns because it is kept in a process-wide registry."""

    def __init__(self, name: str, inputs: dict, stances: bool, source: dict | None = None, baselines: bool = False):
        self.name, self.inputs, self.stances, self.source, self.baselines = name, inputs, stances, source or {}, baselines
        self.t0, self.t1 = time.time(), None
        self.status, self.error, self.result = "running", None, None
        self.agents_done, self.ends, self.node, self.lines = 0, {}, "starting", []

    def on_step(self, state: dict):
        tr = state.get("trace") or []
        if not tr:
            return
        self.node = tr[-1]["node"]
        n = LAST_NODE.get(self.node)
        if n and n > self.agents_done:
            self.agents_done = n
            self.ends[n] = time.time() - self.t0

    def run(self):
        try:
            from prerun_demo import make_bundle
            b = make_bundle(self.inputs, self.name, use_cached=False, with_stances=self.stances, log=self.lines.append, on_step=self.on_step,
                            with_baselines=self.baselines)
            b["source"] = self.source
            DEMO_DIR.mkdir(exist_ok=True)
            (DEMO_DIR / f"{self.name}.json").write_text(json.dumps(b, indent=1, default=str), encoding="utf8")
            self.result, self.status = b, "done"
        except Exception as e:  # show the reason, not a stack trace
            self.error, self.status = str(e)[:700], "error"
        finally:
            self.t1 = time.time()


@st.cache_resource
def job_registry() -> dict:
    return {}


def preflight() -> str | None:
    """Return a message if the local model server is not ready, so the run fails fast instead of looking stuck."""
    from eia.config import S
    if S.llm_provider != "ollama":
        return None
    import requests
    try:
        names = [m["name"] for m in requests.get(S.ollama_base_url + "/api/tags", timeout=4).json().get("models", [])]
    except Exception:
        return f"Ollama is not reachable at {S.ollama_base_url}. Start the Ollama app, then try again."
    for need in (S.llm_model, S.embed_model):
        if need not in names and f"{need}:latest" not in names:
            return f"The model '{need}' is not installed. Run: ollama pull {need}"
    return None


@st.cache_resource
def warm_up_once() -> bool:
    """Load the models into memory in the background and keep them there, so the first agent does not pay the load time."""
    from eia.config import S
    if S.llm_provider != "ollama":
        return False

    def go():
        try:
            import requests
            requests.post(S.ollama_base_url + "/api/generate", json={"model": S.llm_model, "prompt": "", "keep_alive": "30m"}, timeout=600)
            requests.post(S.ollama_base_url + "/api/embed", json={"model": S.embed_model, "input": "warm up", "keep_alive": "30m"}, timeout=600)
        except Exception:
            pass
    threading.Thread(target=go, daemon=True).start()
    return True


def safe_name(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem)[:40] or "run"


def find_saved(filename: str, sha: str):
    """A saved run of this same file (matched by content hash; older saves without a hash are matched by file name)."""
    if not DEMO_DIR.exists():
        return None
    for p in sorted(DEMO_DIR.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf8"))
        except Exception:
            continue
        src = d.get("source") or {}
        if src.get("sha256") == sha or (not src and p.stem == safe_name(filename)):
            return p, d
    return None


def fmt_s(x: float) -> str:
    return f"{int(x // 60)}:{int(x % 60):02d}"


@st.fragment(run_every=1.5)
def progress_panel(key: str):
    job = job_registry().get(key)
    if job is None:
        return
    if job.status != "running":
        st.rerun()  # full rerun: shows the result or the error
    elapsed = time.time() - job.t0
    st.markdown(f"**Running with the local model.** Elapsed {fmt_s(elapsed)}. Typical total on this laptop: about 5 to 6 minutes"
                + (" plus about 2 minutes for the pressure tests" if job.stances else "")
                + (" plus about 3 minutes for the baselines." if job.baselines else "."))
    rows, prev = [], 0.0
    for i, (_, name, desc, typ) in enumerate(STAGES, start=1):
        if i <= job.agents_done:
            state, cls, tm = "Done", "", fmt_s(job.ends[i] - prev)
            prev = job.ends[i]
        elif i == job.agents_done + 1:
            state, cls, tm = "Running", "", f"{fmt_s(max(0.0, elapsed - prev))} (usually ~{typ} s)"
        else:
            state, cls, tm = "Waiting", " wait", ""
        rows.append(f'<div class="sp-step{cls}"><div class="n">{esc(name)}</div><div class="d">{esc(desc)}</div><div class="s"><b>{state}</b>  {esc(tm)}</div></div>')
    if job.stances:
        state = "Running" if job.agents_done == 4 else "Waiting"
        rows.append(f'<div class="sp-step{"" if state == "Running" else " wait"}"><div class="n">Pressure tests</div><div class="d">Judge re-run for the approve and reject pushes</div><div class="s"><b>{state}</b></div></div>')
    if job.baselines:
        started = any(str(x).strip().startswith("baselines:") for x in list(job.lines))
        rows.append(f'<div class="sp-step{"" if started else " wait"}"><div class="n">Baselines</div><div class="d">Baseline and Baseline + rubric answer the same three user messages</div>'
                    f'<div class="s"><b>{"Running" if started else "Waiting"}</b></div></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.caption("A single model call can take one to three minutes, so this list only changes when an agent finishes. You can switch pages and come back; the run continues.")


# ------------------------------------------------------------------ pages
def page_saved():
    page_header("Saved runs", "Cases that were already evaluated with the local model. They open instantly.")
    files = sorted(DEMO_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if DEMO_DIR.exists() else []
    if not files:
        st.info(f"No saved runs in {DEMO_DIR}. Create them with prerun_demo.py or the Evaluate a proposal page.")
        return
    pick = st.selectbox("Case", files, format_func=lambda p: p.stem)
    render_bundle(json.loads(pick.read_text(encoding="utf8")), pick)


def page_evaluate():
    from eia.config import S
    page_header("Evaluate a proposal", "Upload a club proposal. Agent 1 extracts the claims, then Agents 2 to 4 check them against the knowledge base.")
    warm_up_once()
    reg = job_registry()
    sv = st.session_state.get("saved_view")
    if sv:
        d = json.loads(Path(sv).read_text(encoding="utf8"))
        st.info(f"Showing the saved result of an earlier run on this laptop ({d.get('model')}, {fmt_s(d.get('total_wall_seconds') or 0)}). Nothing is recomputed here.")
        if st.button("Back to the upload form"):
            st.session_state.pop("saved_view", None)
            st.rerun()
        render_bundle(d, Path(sv))
        return
    key = st.session_state.get("job_key")
    job = reg.get(key) if key else None

    if job is not None and job.status == "running":
        progress_panel(key)
        return
    if job is not None and job.status == "error":
        st.error(f"The run failed after {fmt_s((job.t1 or time.time()) - job.t0)}: {job.error}")
        if st.button("Try again"):
            st.session_state.pop("job_key", None)
            st.rerun()
        return
    if job is not None and job.status == "done":
        st.success(f"Finished in {fmt_s(job.t1 - job.t0)}. Saved as {job.name}.json in {DEMO_DIR.name}/ so it also appears under Saved runs.")
        if st.button("Evaluate another proposal"):
            st.session_state.pop("job_key", None)
            st.rerun()
        render_bundle(job.result, DEMO_DIR / f"{job.name}.json")
        return

    with st.container(border=True):
        f = st.file_uploader("Proposal file", type=["pdf", "docx", "txt", "md"], help="PDF, DOCX or plain text")
        c1, c2, c3 = st.columns(3, gap="medium")
        st.caption("These three fields are optional. Anything you fill in here overrides what the document says, so leave them as they are to use the document.")
        sub = c1.date_input("Submission date", value=None, help="Leave empty to use the date in the document")
        cap = c2.number_input("Venue capacity", min_value=0, value=0, step=10, help="0 means not given")
        dean = c3.selectbox("Dean approval", ["Use the document", "Yes", "No"])
        stances = st.checkbox("Also run the two pressure tests (about 2 more minutes)", value=False)
        baselines = st.checkbox("Also run the baseline and baseline + rubric for comparison (about 3 more minutes)", value=False)
        sha = hashlib.sha256(f.getvalue()).hexdigest() if f is not None else None
        hit = find_saved(f.name, sha) if f is not None else None
        if hit:
            st.info(f"This file was already evaluated on this laptop ({hit[1].get('model')}, {fmt_s(hit[1].get('total_wall_seconds') or 0)}). You can open that saved result now, or run it again.")
            if st.button("Open the saved result"):
                st.session_state["saved_view"] = str(hit[0])
                st.rerun()
        go = st.button("Run it again" if hit else "Evaluate", type="primary", disabled=f is None or any(j.status == "running" for j in reg.values()))
    st.caption(f"Model: {S.llm_provider} / {S.llm_model}  |  {S.prompt_docs_per_claim} documents per claim  |  {S.snippet_chars}-character snippets  |  timeout {S.llm_timeout} s")
    if go and f is not None:
        problem = preflight()
        if problem:
            st.error(problem)
            return
        tmp = Path(tempfile.mkdtemp()) / f.name
        tmp.write_bytes(f.getvalue())
        form = {k: v for k, v in {"submission_date": str(sub) if sub else None, "venue_capacity": int(cap) or None, "dean_approval": None if dean == "Use the document" else dean}.items() if v is not None}
        name = safe_name(f.name)
        if (DEMO_DIR / f"{name}.json").exists():
            name += time.strftime("_%H%M%S")
        job = Job(name, {"proposal_path": str(tmp), "proposal_form": form}, stances, source={"name": f.name, "sha256": sha}, baselines=baselines)
        key = f"{name}-{int(job.t0)}"
        reg[key] = job
        st.session_state["job_key"] = key
        threading.Thread(target=job.run, daemon=True).start()
        st.rerun()


def page_results():
    page_header("Results", "The experiment: baseline, baseline with the rubric, and the framework, each asked three ways (neutral, push to approve, push to reject).")
    sj = RESULTS_DIR / "summary.json"
    if not sj.exists():
        st.info(f"No evaluation results found in {RESULTS_DIR}. Run evaluate.py, then analyze.py, or point EIA_RESULTS_DIR at the results folder.")
        return
    res = json.loads(sj.read_text(encoding="utf8"))
    n = max(v.get("n_proposals", 0) for v in res.values())
    model = None
    mf = RESULTS_DIR / "manifest.json"
    if mf.exists():
        try:
            m = json.loads(mf.read_text(encoding="utf8"))
            model = m.get("model") or (m.get("manifests") or [{}])[0].get("model")
        except Exception:
            pass
    st.caption(f"{n} test proposals" + (f"  |  model {model}" if model else "") + f"  |  folder {RESULTS_DIR}")

    fw, base = res.get("framework"), res.get("baseline")
    if fw and base:
        cols = st.columns(4, gap="medium")
        pts = lambda k: (fw[k] - base[k]) * 100
        cols[0].metric("Human agreement", f"{fw['human_agreement'] * 100:.1f}%", f"{pts('human_agreement'):+.1f} pts vs baseline")
        cols[1].metric("Agreement bias score", f"{fw['agreement_bias_score'] * 100:.1f}%", f"{pts('agreement_bias_score'):+.1f} pts vs baseline", delta_color="inverse")
        cols[2].metric("Decision stability", f"{fw['decision_stability'] * 100:.1f}%", f"{pts('decision_stability'):+.1f} pts vs baseline")
        cols[3].metric("Calibration error", f"{fw['calibration_error_mean_abs_pct']:.1f} pts", f"{fw['calibration_error_mean_abs_pct'] - base['calibration_error_mean_abs_pct']:+.1f} vs baseline", delta_color="inverse")
        st.caption("Framework compared with the plain baseline. Agreement bias is how often the answer moved toward what the user pushed for (lower is better).")

    df = pd.DataFrame(res).T
    pct = ["human_agreement", "agreement_bias_score", "decision_flip_rate", "decision_stability"]
    show = {"human_agreement": "Human agreement %", "agreement_bias_score": "Agreement bias %", "decision_flip_rate": "Flip rate %",
            "decision_stability": "Stability %", "calibration_error_mean_abs_pct": "Calibration error", "evidence_diversity_mean_docs": "Evidence docs/run",
            "latency_mean_seconds_per_run": "Latency s/run", "n_runs": "Runs", "n_proposals": "Proposals"}
    tab = df[[c for c in show if c in df.columns]].copy()
    for c in pct:
        if c in tab:
            tab[c] = (tab[c].astype(float) * 100).round(1)
    st.dataframe(tab.rename(columns=show), width="stretch")

    import altair as alt
    sysorder = [x for x in ("baseline", "baseline_rubric", "framework") if x in df.index]
    c1, c2 = st.columns(2, gap="large")
    long = pd.DataFrame({"system": sysorder * 2, "metric": ["Human agreement"] * len(sysorder) + ["Agreement bias score"] * len(sysorder),
                         "value": [df.loc[x, "human_agreement"] * 100 for x in sysorder] + [df.loc[x, "agreement_bias_score"] * 100 for x in sysorder]})
    c1.markdown("**Agreement and agreement bias** (%)")
    c1.altair_chart(alt.Chart(long).mark_bar().encode(
        x=alt.X("system:N", sort=sysorder, title=None, axis=alt.Axis(labelAngle=0)), xOffset="metric:N", y=alt.Y("value:Q", title="%", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("metric:N", title=None, scale=alt.Scale(domain=["Human agreement", "Agreement bias score"], range=["#1b9e6b", "#cf3b3b"]), legend=alt.Legend(orient="bottom")),
        tooltip=["system", "metric", alt.Tooltip("value:Q", format=".1f")]).properties(height=300), width="stretch")
    ec = RESULTS_DIR / "eval_runs.csv"
    if ec.exists():
        runs = pd.read_csv(ec)
        cnt = runs.groupby(["system", "decision"]).size().reset_index(name="runs")
        order = ["Approve", "Approve with Conditions", "Needs More Information", "Reject"]
        c2.markdown("**Decisions given** (runs per system)")
        c2.altair_chart(alt.Chart(cnt).mark_bar().encode(
            x=alt.X("system:N", sort=sysorder, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("runs:Q", title="runs"),
            color=alt.Color("decision:N", title=None, sort=order, scale=alt.Scale(domain=order, range=["#1b9e6b", "#3b6fd4", "#d28a0c", "#cf3b3b"]), legend=alt.Legend(orient="bottom", columns=2)),
            order=alt.Order("decision:N", sort="ascending"),
            tooltip=["system", "decision", "runs"]).properties(height=300), width="stretch")
    st.info("Read with care: the confidence intervals and significance tests are in the full report below, and the per-scenario breakdowns have only a few proposals in each cell.")
    rp = RESULTS_DIR / "report.md"
    if rp.exists():
        with st.expander("Full report (tables, confidence intervals, significance tests)"):
            st.markdown(rp.read_text(encoding="utf8"))


# ------------------------------------------------------------------ shell: brand top left, navigation on top
st.session_state.setdefault("page", PAGES[0])
st.session_state.setdefault("nav", st.session_state["page"])


def _on_nav():
    v = st.session_state.get("nav")
    if v:
        st.session_state["page"] = v
    else:  # clicking the selected item would clear it; keep the current page selected
        st.session_state["nav"] = st.session_state["page"]


brand, nav = st.columns([1.1, 2.4], vertical_alignment="center", gap="large")
brand.markdown('<div class="sp-brand">Spectrum<span>Anti-sycophancy decision support</span></div>', unsafe_allow_html=True)
with nav:
    st.segmented_control("Navigation", PAGES, key="nav", on_change=_on_nav, label_visibility="collapsed")
st.markdown('<hr class="sp-rule">', unsafe_allow_html=True)

{"Saved runs": page_saved, "Evaluate a proposal": page_evaluate, "Results": page_results}[st.session_state["page"]]()
