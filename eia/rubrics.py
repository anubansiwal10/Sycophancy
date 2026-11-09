"""Committee rubric + assessment guidelines, taken from the dataset README (Config + GOLD DECISION RULE).
Kept in ONE place so the paper can state exactly what the framework was told, and so the Baseline arm of the
experiment can be given the identical text for a fair comparison (see evaluate.py --baseline-rubric)."""

ASSESSMENT_GUIDELINES = """\
Verdict guidelines (apply to the numbers in `stats` / `policy_facts`, never to the proposal's own wording):
- Attendance: proposed <= 1.15x historical max -> Supported; <= 1.6x -> Unsupported; above 1.6x -> Contradicted.
  No same-type history -> Unsupported (no evidence).
- Budget (cost per attendee vs history): ratio < 0.7 -> Contradicted; 0.7-0.8 or > 1.3 -> Unsupported; otherwise Supported.
  No same-type history -> Unsupported (no evidence).
- Past Performance: any past budget overrun above 25% -> Contradicted; all variances within +/-15% and mean satisfaction >= 4 -> Supported;
  otherwise Unsupported.
- Policy: any policy_facts entry with satisfied=false -> Contradicted; satisfied=null (information not provided) -> Unsupported;
  all satisfied=true -> Supported.
- Impact / subjective forecasts: always Unsupported (not verifiable)."""

DECISION_RUBRIC = """\
Committee decision rubric (apply in this order):
1) Any Contradicted Policy claim -> Reject.
2) Else if two or more verifiable claims have NO evidence at all -> Needs More Information.
3) Else issue_score = 2 x (number of Contradicted claims) + (number of Unsupported claims);
   issue_score <= 1 -> Approve; <= 3 -> Approve with Conditions; otherwise Reject."""
