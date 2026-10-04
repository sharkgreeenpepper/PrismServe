# Retrospective answer-selection diagnostic

**Date:** 2026-10-04 23:53 Asia/Shanghai  
**Input:** B1.6 terminal-budget run and its paired direct-answer outputs.  
**Purpose:** Check whether the failed exact-match screen is readily explained by the existing all-node vote. This is a post-hoc sensitivity analysis on an already-used cohort, not a confirmatory experiment or a new policy result.

## Results

| Retrospective selection rule | Numeric-answer coverage | Exact match |
|---|---:|---:|
| Existing plurality over every completed node | 16/16 | 5/16 |
| Plurality over completed nodes with no released children | 16/16 | 5/16 |
| One vote per first-level branch: take the deepest answer-bearing nodes in each branch, vote within the branch, then vote across branches | 11/16 | 3/16 |
| Plurality over forced-final nodes only | 11/16 | 3/16 |
| Add the paired direct answer as one more vote to the existing all-node pool | 16/16 | 5/16 |
| Paired direct answer alone | 16/16 | 8/16 |

For the branch-balanced rule, a tree whose root itself returned a final answer has no first-level branch vote; the 11/16 coverage reflects that definition and is not a deployable fallback policy. The forced-final-only rule also has no candidate for five questions. Both are diagnostics of where candidate answers came from.

Adding the direct answer to the all-node vote pool did not repair any tree-wrong answer and changed three direct-correct questions (dataset indices 755, 347, and 434) to incorrect pooled answers. The paired direct result is therefore a useful quality reference, but simply adding it as another equal vote is not a suitable selector.

## Interpretation and decision

Leaf-only voting matched the existing all-node result, so ancestor/descendant duplicate votes do not explain the observed 5/16 exact-match rate on this cohort. The tested branch-balanced and forced-final-only summaries were lower and had incomplete coverage. These post-hoc checks do not establish that another answer selector cannot work; they show that these simple alternatives do not justify reopening the quality gate.

Keep R004/R005 stopped. Do not use these already-seen questions to tune a selector or make a scheduler comparison. Reopening requires a materially different, fully specified online search/answer-selection policy and a fresh untouched direct-reference cohort that passes the frozen quality gate. Any verifier-based selector would need its extra calls, latency, and token cost counted in that gate.

## Method and provenance

- Source tree events: `/home/bumi/infra/cache/direction1-terminal-budget-quality-20261004/tree/terminal-budget-tree-20261004-*_EVENTS.jsonl` and `terminal-budget-tree-20261004_TREES.jsonl`.
- Paired direct records: `/home/bumi/infra/cache/direction1-terminal-budget-quality-20261004/direct/terminal-budget-direct-20261004_RECORDS.jsonl`.
- The answer parser and plurality tie-break follow `run_live_tree_smoke.py`: parse numeric `status=final` outputs, normalize numeric strings, and break ties lexicographically.
- The cohort has 16 questions and was already used for B1.6. Counts are descriptive; no uncertainty interval or inferential claim is made.
- Primary B1.6 quality report: `LIVE_TREE_TERMINAL_BUDGET_VALIDATION_REPORT.md`.
