"""Merge result folders from several machines into one.

  python merge_results.py results_local_anu results_local_teammate --out results_local --check PRP-0864

Refuses to merge if the two runs used different code, model or settings, or if any proposal appears in more than one
folder (except the --check proposal, which both machines run on purpose; its two copies are compared and the first
folder's copy is kept).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

KEYS = ["git_commit", "model", "llm_provider", "embed_provider", "seed", "n_proposals", "split", "cached_analysis", "settings"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", default="results_local")
    ap.add_argument("--check", default="", help="proposal id run on every machine (cross-machine consistency check)")
    a = ap.parse_args()
    check = {x.strip() for x in a.check.split(",") if x.strip()}

    mans, rows = [], []
    for f in a.folders:
        d = Path(f)
        mans.append(json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {})
        for line in (d / "eval_runs.jsonl").read_text().splitlines():
            r = json.loads(line); r["_src"] = f; rows.append(r)
    problems = []
    for k in KEYS:
        vals = {json.dumps(m.get(k), sort_keys=True) for m in mans}
        if len(vals) > 1:
            problems.append(f"manifests differ on '{k}': {sorted(vals)}")
    df = pd.DataFrame(rows)
    owners = df.groupby("proposal_id")["_src"].nunique()
    clash = [p for p, n in owners.items() if n > 1 and p not in check]
    if clash:
        problems.append(f"proposals present in more than one folder: {clash}")
    if problems:
        print("NOT MERGING:"); [print(" -", p) for p in problems]; raise SystemExit(1)

    for pid in sorted(check):
        sub = df[(df.proposal_id == pid) & (df.system == "framework")]
        if sub["_src"].nunique() < 2:
            print(f"check {pid}: only one machine has it"); continue
        piv = sub.pivot_table(index="stance", columns="_src", values="decision", aggfunc="first")
        conf = sub.pivot_table(index="stance", columns="_src", values="confidence_pct", aggfunc="first")
        print(f"check {pid}: decisions per machine\n{piv}\nconfidence per machine\n{conf}")
        print("  -> SAME decisions on both machines" if piv.nunique(axis=1).max() == 1 else "  -> decisions DIFFER (LLM nondeterminism or setup mismatch; report it)")

    first = a.folders[0]
    keep = df[~((df.proposal_id.isin(check)) & (df["_src"] != first))].drop(columns="_src")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with (out / "eval_runs.jsonl").open("w") as f:
        for r in keep.to_dict("records"):
            f.write(json.dumps(r, default=str) + "\n")
    fails = []
    for f_ in a.folders:
        p = Path(f_) / "failures.jsonl"
        if p.exists():
            fails += p.read_text().splitlines()
    (out / "failures.jsonl").write_text("\n".join(fails) + ("\n" if fails else ""))
    (out / "manifest.json").write_text(json.dumps({"merged_from": a.folders, "manifests": mans, "check": sorted(check)}, indent=1))
    print(f"merged {len(keep)} runs, {keep.proposal_id.nunique()} proposals -> {out}/ ; failures logged: {len(fails)}")


if __name__ == "__main__":
    main()
