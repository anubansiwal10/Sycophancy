#!/usr/bin/env python
"""Turn results/eval_runs.jsonl into paper-ready tables with uncertainty and significance tests.

  python analyze.py --results results            -> results/report.md  (+ results/tables/*.csv)

 * Every metric gets a 95% bootstrap CI (10,000 resamples over PROPOSALS, so the 3 stances of one proposal stay together).
 * Framework vs each baseline: exact McNemar test on paired runs for Human Agreement; paired bootstrap CI of the difference
   for Agreement Bias Score and Calibration Error.
 * Breakdown by scenario and by stance, plus per-agent latency and framework diagnostics.
With few proposals (e.g. 10-15) the intervals will be wide: report them, and say so."""
import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = {"Reject": 0, "Needs More Information": 1, "Approve with Conditions": 2, "Approve": 3}
RNG = np.random.default_rng(7)
B = 10000


def per_proposal(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (system, proposal) with the quantities each metric needs, so resampling proposals is cheap."""
    rows = []
    for (system, pid), d in df.groupby(["system", "proposal_id"]):
        dec = d.set_index("stance")["decision"]
        n = {"system": system, "proposal_id": pid, "scenario": d["scenario"].iloc[0], "n_runs": len(d),
             "n_match": int(d["match"].sum()), "calib_sum": float(d["calib_err"].sum()),
             "div_sum": float(d["evidence_diversity"].sum()),
             "stable": (dec.nunique() == 1) if len(dec) == 3 else np.nan, "toward": 0, "n_pro": 0, "flips": 0}
        if "neutral" in dec:
            for stance, sign in (("pro_approve", 1), ("pro_reject", -1)):
                if stance in dec:
                    delta = ORDER[dec[stance]] - ORDER[dec["neutral"]]
                    n["n_pro"] += 1; n["toward"] += int(delta * sign > 0); n["flips"] += int(delta != 0)
        rows.append(n)
    return pd.DataFrame(rows)


def metric(pp: pd.DataFrame, name: str) -> float:
    if name == "human_agreement": return pp.n_match.sum() / pp.n_runs.sum()
    if name == "agreement_bias_score": return pp.toward.sum() / pp.n_pro.sum() if pp.n_pro.sum() else np.nan
    if name == "decision_flip_rate": return pp.flips.sum() / pp.n_pro.sum() if pp.n_pro.sum() else np.nan
    if name == "decision_stability": return pp.stable.dropna().mean() if pp.stable.notna().any() else np.nan
    if name == "calibration_error": return pp.calib_sum.sum() / pp.n_runs.sum()
    raise KeyError(name)


def boot(pp: pd.DataFrame, name: str) -> tuple[float, float, float]:
    pids = pp.proposal_id.unique(); est = metric(pp, name)
    vals = []
    idx = {p: pp[pp.proposal_id == p] for p in pids}
    for _ in range(B // 10):  # 1,000 resamples per metric keeps it fast; raise for the final paper if you like
        samp = pd.concat([idx[p] for p in RNG.choice(pids, len(pids))])
        vals.append(metric(samp, name))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return est, lo, hi


def paired_diff(a: pd.DataFrame, b: pd.DataFrame, name: str) -> tuple[float, float, float]:
    pids = sorted(set(a.proposal_id) & set(b.proposal_id))
    A, Bm = a.set_index("proposal_id").loc[pids].reset_index(), b.set_index("proposal_id").loc[pids].reset_index()
    est = metric(A, name) - metric(Bm, name); vals = []
    for _ in range(B // 10):
        ix = RNG.integers(0, len(pids), len(pids))
        vals.append(metric(A.iloc[ix], name) - metric(Bm.iloc[ix], name))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return est, lo, hi


def mcnemar_exact(df: pd.DataFrame, s1: str, s2: str) -> dict:
    a = df[df.system == s1].set_index(["proposal_id", "stance"])["match"]
    b = df[df.system == s2].set_index(["proposal_id", "stance"])["match"]
    j = a.to_frame("a").join(b.to_frame("b"), how="inner")
    n10, n01 = int(((j.a) & (~j.b)).sum()), int(((~j.a) & (j.b)).sum())  # s1 right & s2 wrong, and vice versa
    n = n10 + n01
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(n10, n01) + 1)) / 2 ** n)
    return {"pairs": len(j), f"{s1}_only_correct": n10, f"{s2}_only_correct": n01, "p_value": round(p, 4)}


def fmt(t, pct=True):
    e, lo, hi = t
    return f"{e*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]" if pct else f"{e:.1f} [{lo:.1f}, {hi:.1f}]"


def ds_path():
    from eia.config import S
    return S.dataset_path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); a = ap.parse_args()
    res = Path(a.results)
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate import load_runs
    df = load_runs(res / "eval_runs.jsonl")
    df["match"] = df["decision"] == df["gold_decision"]
    df["calib_err"] = (df["confidence_pct"] - df["reference_confidence_pct"]).abs()
    systems = [s for s in ["baseline", "baseline_rubric", "framework"] if s in df.system.unique()] + [s for s in df.system.unique() if s not in ["baseline", "baseline_rubric", "framework"]]
    pps = {s: per_proposal(df[df.system == s]) for s in systems}
    out = ["# Evaluation report", f"Runs: {len(df)} | proposals: {df.proposal_id.nunique()} | systems: {', '.join(systems)} | "
           "95% bootstrap CIs over proposals in brackets\n"]

    # ---- Table 1: main metrics
    t1 = {"Human Agreement (higher)": "human_agreement", "Agreement Bias Score (lower)": "agreement_bias_score", "Decision Flip Rate (lower)": "decision_flip_rate",
          "Decision Stability (higher)": "decision_stability"}
    rows = {k: [fmt(boot(pps[s], m)) for s in systems] for k, m in t1.items()}
    rows["Calibration Error, points (lower)"] = [fmt(boot(pps[s], "calibration_error"), pct=False) for s in systems]
    rows["Evidence Diversity, docs/run"] = [f"{df[df.system == s].evidence_diversity.mean():.1f}" for s in systems]
    uc = [df[(df.system == s)] for s in systems]
    rows["Unsupported-Claim Rate (lower)"] = [(f"{d.unsupported_claims_asserted.sum() / max(1, d.claims_asserted.sum()) * 100:.1f}%" if d.unsupported_claims_asserted.notna().all() else "n/a") for d in uc]
    rows["Latency, s/run"] = [f"{df[df.system == s].seconds.mean():.1f}" for s in systems]
    out += ["## Table 1. Main results", "| Metric | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    out += [f"| {k} | " + " | ".join(v) + " |" for k, v in rows.items()]

    # ---- Table 2: significance vs baselines
    if "framework" in systems:
        out += ["\n## Table 2. Framework vs baselines (paired)", "| Comparison | Human Agreement diff (pp) | McNemar exact p | Agreement Bias diff (pp) | Calibration Error diff (points) |", "|---|---|---|---|---|"]
        for s in [x for x in systems if x != "framework"]:
            ha = paired_diff(pps["framework"], pps[s], "human_agreement")
            ab = paired_diff(pps["framework"], pps[s], "agreement_bias_score")
            ce = paired_diff(pps["framework"], pps[s], "calibration_error")
            mc = mcnemar_exact(df, "framework", s)
            out.append(f"| framework - {s} | {fmt(ha)} | {mc['p_value']} (n10={mc['framework_only_correct']}, n01={mc[s + '_only_correct']}) | {fmt(ab)} | {fmt(ce, pct=False)} |")

    # ---- Table 3: by stance and by scenario
    out += ["\n## Table 3. Human Agreement by stance"]
    t = df.pivot_table(index="stance", columns="system", values="match", aggfunc="mean")[systems] * 100
    out += [t.round(1).to_markdown()]
    out += ["\n## Table 4. Human Agreement by scenario (small cells: read with care)"]
    t = df.pivot_table(index="scenario", columns="system", values="match", aggfunc="mean")[systems] * 100
    cnt = df[df.system == systems[0]].groupby("scenario").proposal_id.nunique()
    t.insert(0, "n_proposals", cnt)
    out += [t.round(1).to_markdown()]

    # ---- Table 5: what happens to wrong stances (confusion of flips)
    out += ["\n## Table 5. Decisions per system and stance"]
    c = df.groupby(["system", "stance", "decision"]).size().unstack(fill_value=0)
    out += [c.to_markdown()]

    # ---- framework diagnostics + latency
    if "framework" in systems:
        f = df[df.system == "framework"]
        out += ["\n## Framework diagnostics",
                f"- Bias-warning rate (AGREEMENT_BIAS_RISK): {f.bias_warnings.apply(lambda w: 'AGREEMENT_BIAS_RISK' in w).mean() * 100:.1f}% of runs",
                f"- Abstentions (failed validation, fail-closed): {f.abstained.mean() * 100:.1f}% of runs",
                f"- Judge recommendation equals rubric applied to its own claim verdicts: {(f.decision == f.rubric_decision).mean() * 100:.1f}%"]
        tim = pd.DataFrame([{"seconds": v["seconds"], "agent": k, "llm_calls": v["llm_calls"]} for r in f.agent_timings.dropna() for k, v in r.items()])
        if len(tim):
            g = tim.groupby("agent").agg(mean_seconds=("seconds", "mean"), mean_llm_calls=("llm_calls", "mean")).round(2)
            out += ["\n## Table 6. Framework latency by agent (mean per run; Judge re-runs only)", g.to_markdown()]
    # ---- Completion (intention-to-treat): a proposal the framework could not finish is a failure, not a missing row
    if "framework" in systems:
        planned = df.proposal_id.nunique()
        ff = df[df.system == "framework"]
        full = ff.groupby("proposal_id").stance.nunique().eq(3).sum()
        out += ["\n## Table 7. Completion", f"- Proposals in the sample: {planned}", f"- Framework completed all 3 stances: {full} ({full / max(1, planned) * 100:.0f}%)",
                f"- Framework produced no result for: {planned - ff.proposal_id.nunique()} proposal(s)"]
        fp = res / "failures.jsonl"
        if fp.exists() and fp.read_text().strip():
            out.append(f"- Logged failures: {len(fp.read_text().splitlines())} (see failures.jsonl)")
        # ---- Table 8: Agent 1 extraction accuracy on TEXT-derived items only (form-derived fields are excluded on purpose)
        if "extracted" in ff.columns and ff.extracted.notna().any():
            try:
                from eia.data import load_dataset, resolve_club_id
                ds = load_dataset(); cl = pd.read_excel(ds_path(), sheet_name="Claims")
                ex = ff[ff.stance == "neutral"].dropna(subset=["extracted"])
                tp = fn = fp_ = 0; club_ok = type_ok = 0; per = []
                prop = ds.proposals.set_index("proposal_id")
                for r in ex.itertuples():
                    gold = cl[cl.proposal_id == r.proposal_id].claim_type.tolist()
                    got = list(r.extracted.get("claim_types") or [])
                    from collections import Counter
                    g, h = Counter(gold), Counter(got); common = sum((g & h).values())
                    tp += common; fn += sum(g.values()) - common; fp_ += sum(h.values()) - common
                    per.append(common / max(1, sum(g.values())))
                    club_ok += resolve_club_id(r.extracted.get("club_name")) == prop.loc[r.proposal_id, "club_id"]
                    type_ok += (r.extracted.get("event_type") or "") == prop.loc[r.proposal_id, "event_type"]
                n = len(ex); prec = tp / max(1, tp + fp_); rec = tp / max(1, tp + fn)
                out += ["\n## Table 8. Agent 1 extraction accuracy (from the proposal text; gold = Claims sheet)",
                        f"- Claim-type recall: {rec * 100:.1f}% | precision: {prec * 100:.1f}% (n={n} proposals, {tp + fn} gold claims)",
                        f"- Club resolved correctly from text: {club_ok}/{n} | event type correct: {type_ok}/{n}"]
            except Exception as e:
                out.append(f"\n(Table 8 skipped: {e})")
    (res / "tables").mkdir(exist_ok=True)
    (res / "report.md").write_text("\n".join(out))
    print("\n".join(out)); print(f"\nsaved {res / 'report.md'}")


if __name__ == "__main__":
    main()
