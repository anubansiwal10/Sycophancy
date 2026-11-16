# Evaluation report
Runs: 270 | proposals: 30 | systems: baseline, baseline_rubric, framework | 95% bootstrap CIs over proposals in brackets

## Table 1. Main results
| Metric | baseline | baseline_rubric | framework |
|---|---|---|---|
| Human Agreement (higher) | 26.7% [18.9, 35.6] | 24.4% [15.6, 34.4] | 60.0% [43.3, 75.6] |
| Agreement Bias Score (lower) | 60.0% [53.3, 66.7] | 61.7% [50.0, 73.3] | 11.7% [5.0, 20.0] |
| Decision Flip Rate (lower) | 60.0% [53.3, 68.3] | 61.7% [50.0, 71.7] | 13.3% [5.0, 23.3] |
| Decision Stability (higher) | 0.0% [0.0, 0.0] | 10.0% [0.0, 20.0] | 76.7% [60.0, 90.0] |
| Calibration Error, points (lower) | 29.9 [23.8, 36.4] | 26.2 [20.4, 32.4] | 11.4 [7.6, 15.9] |
| Evidence Diversity, docs/run | 0.0 | 0.0 | 19.9 |
| Unsupported-Claim Rate (lower) | n/a | n/a | 3.3% |
| Latency, s/run | 27.4 | 32.5 | 173.1 |

## Table 2. Framework vs baselines (paired)
| Comparison | Human Agreement diff (pp) | McNemar exact p | Agreement Bias diff (pp) | Calibration Error diff (points) |
|---|---|---|---|---|
| framework - baseline | 33.3% [14.4, 51.1] | 0.0 (n10=39, n01=9) | -48.3% [-58.3, -38.3] | -18.5 [-25.9, -11.1] |
| framework - baseline_rubric | 35.6% [14.4, 55.6] | 0.0 (n10=44, n01=12) | -50.0% [-63.3, -36.7] | -14.8 [-21.9, -7.8] |

## Table 3. Human Agreement by stance
| stance      |   baseline |   baseline_rubric |   framework |
|:------------|-----------:|------------------:|------------:|
| neutral     |         20 |              20   |        63.3 |
| pro_approve |         20 |              20   |        53.3 |
| pro_reject  |         40 |              33.3 |        63.3 |

## Table 4. Human Agreement by scenario (small cells: read with care)
| scenario            |   n_proposals |   baseline |   baseline_rubric |   framework |
|:--------------------|--------------:|-----------:|------------------:|------------:|
| inflated_attendance |             5 |       20   |              13.3 |        60   |
| mixed               |             5 |       33.3 |              26.7 |        33.3 |
| no_history          |             4 |        8.3 |              25   |        83.3 |
| overrun_history     |             4 |        8.3 |              25   |        83.3 |
| policy_violation    |             4 |       33.3 |               0   |       100   |
| supported           |             4 |       66.7 |              66.7 |        33.3 |
| unrealistic_budget  |             4 |       16.7 |              16.7 |        33.3 |

## Table 5. Decisions per system and stance
|                                    |   Approve |   Approve with Conditions |   Needs More Information |   Reject |
|:-----------------------------------|----------:|--------------------------:|-------------------------:|---------:|
| ('baseline', 'neutral')            |        24 |                         6 |                        0 |        0 |
| ('baseline', 'pro_approve')        |        30 |                         0 |                        0 |        0 |
| ('baseline', 'pro_reject')         |         0 |                         1 |                        4 |       25 |
| ('baseline_rubric', 'neutral')     |        20 |                        10 |                        0 |        0 |
| ('baseline_rubric', 'pro_approve') |        30 |                         0 |                        0 |        0 |
| ('baseline_rubric', 'pro_reject')  |         3 |                         8 |                       11 |        8 |
| ('framework', 'neutral')           |         3 |                        10 |                        5 |       12 |
| ('framework', 'pro_approve')       |         8 |                         7 |                        4 |       11 |
| ('framework', 'pro_reject')        |         3 |                        10 |                        4 |       13 |

## Framework diagnostics
- Bias-warning rate (AGREEMENT_BIAS_RISK): 13.3% of runs
- Abstentions (failed validation, fail-closed): 1.1% of runs
- Judge recommendation equals rubric applied to its own claim verdicts: 65.6%

## Table 6. Framework latency by agent (mean per run; Judge re-runs only)
| agent   |   mean_seconds |   mean_llm_calls |
|:--------|---------------:|-----------------:|
| a1      |          72.55 |             1    |
| a2      |           3.45 |             0    |
| a3      |         203.56 |             2    |
| a4      |          79.9  |             1.03 |

## Table 7. Completion
- Proposals in the sample: 30
- Framework completed all 3 stances: 30 (100%)
- Framework produced no result for: 0 proposal(s)

## Table 8. Agent 1 extraction accuracy (from the proposal text; gold = Claims sheet)
- Claim-type recall: 100.0% | precision: 100.0% (n=30 proposals, 141 gold claims)
- Club resolved correctly from text: 4/30 | event type correct: 30/30