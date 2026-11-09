# Spectrum demo structured proposals

These three JSON files use the Agent-1 bypass schema expected by `run.py --analysis-json`.

## Files
1. `01_badminton_capacity_problem.json` — attendance claim conflicts with historical attendance and exceeds Seminar Hall capacity.
2. `02_dance_budget_overrun_problem.json` — budget is far below historical per-attendee costs; prior event overruns make the omitted variance justification relevant.
3. `03_music_society_strong_proposal.json` — designed to align with Music Society historical Cultural Fest attendance, spending, venue capacity, and policy timing.

## Where to store them
Recommended: create `demo_proposals` inside your project root (the same folder containing `run.py`) and put all three JSON files there.

Example:
`eia_project/demo_proposals/01_badminton_capacity_problem.json`

## Run from the project root in PowerShell
```powershell
python run.py --analysis-json demo_proposals/01_badminton_capacity_problem.json
python run.py --analysis-json demo_proposals/02_dance_budget_overrun_problem.json
python run.py --analysis-json demo_proposals/03_music_society_strong_proposal.json
```

Do not add `--cached`: `--analysis-json` already selects the Agent-1 bypass for your supplied structured JSON, while Agents 2–4 still run.

Each run should create a full result under `runs/` and per-agent JSON files under `runs/run_<run-id>_agents/`.

## Important
- These are synthetic demo proposals matched to clubs and historical records in the uploaded synthetic dataset.
- They do not contain gold labels or copied evidence verdicts. Retrieval and review must produce those from the project's knowledge base.
- Ensure the vector index was built from the same dataset version (`python build_index.py`) before the demo.
- The `club_id` and `club_name` must match the dataset.
