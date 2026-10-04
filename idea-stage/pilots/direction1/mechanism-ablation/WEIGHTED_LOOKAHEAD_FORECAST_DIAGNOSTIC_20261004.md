# Offline remaining-work estimate diagnostic — 2026-10-04

**Status:** exploratory, post-hoc analysis of the completed block-A request records; no GPU was used. This does not alter the frozen screen or its `claim_supported: no` verdict.

## Method

For each tree root, the router records `predicted_remaining_tree_work_s` as the estimated service work of all descendants if they stay on the selected worker. I compared it with the sum of descendant vLLM time-to-first-token plus decode time. vLLM documents TTFT as scheduling-to-first-token and generation time as decode from first to last token, excluding queue wait; queue time is therefore reported separately. See the [vLLM 0.30 per-request metrics documentation](https://docs.vllm.ai/en/v0.30.0/features/per_request_metrics/).

This is a descriptive calibration check over 16 fixed trees. Descendant requests overlap in time, so their summed queue wait and HTTP latency are cumulative request measures, not tree wall time. The router estimate assumes descendants remain on one worker while the actual policy may distribute them.

## Results

| Arm | Median predicted remaining work (s) | Median realized descendant TTFT + decode (s) | Median realized / predicted | Median absolute relative error | Spearman rank correlation | Median summed descendant queue wait (s) |
|---|---:|---:|---:|---:|---:|---:|
| flat | 93.870 | 80.214 | 0.958 | 12.97% | 0.568 | 85.859 |
| tree_w1.0 | 93.870 | 80.252 | 0.942 | 18.49% | 0.574 | 81.676 |
| tree_w0.5 | 93.870 | 83.379 | 0.884 | 15.99% | 0.629 | 79.543 |

The stored root estimates were identical across all three arms for each tree (maximum within-tree difference 0.000000 s). The median absolute prediction error ranged from 12.97% to 18.49%, and Spearman rank correlation ranged from 0.568 to 0.629. Weight 0.5 did not improve this descriptive calibration measure over the other arms.

Queue totals are substantial in the request records, but summing waits across overlapping descendants does not imply the tree would finish that much later. The comparison suggests that the scalar lookahead weight is not the only uncertainty: estimated descendant service varies from realized work by tree, and queueing adds a separate large per-request component. One 16-tree block is insufficient to tune either term or establish causality.

## Held-out output-length profile check

The run's output-length profile came from four earlier calibration trees (16 inner-node and 10 leaf observations). Their eight GSM8K dataset indices were disjoint from the 16 questions in this screen. In the flat arm, the profile expected 529 inner tokens and 532.5 leaf tokens; the new run observed 68 inner requests at mean/median 511.9/485 tokens and 60 leaf requests at mean/median 391.7/355 tokens. The leaf mean was 26.4% below the prior and the leaf median 33.3% below it. The tree arms showed a similar leaf gap, and no leaf request ended at the token limit.

This is out-of-sample evidence that the small leaf-length prior was high for this cohort. It is a plausible source of remaining-work estimate error, but it does not establish that the prior caused the routing or latency outcome. The calibration sample is small and output lengths still vary by question and run.

## Next diagnostic

Before any new GPU allocation, recalibrate inner and leaf output-work distributions on a larger, shape-stratified calibration sample, separate output-length error from per-worker service-rate error, and inspect whether queue pressure belongs as a distinct route-objective term. Keep the current 16-question cohort as calibration evidence only; evaluate any candidate on a new independent cohort with counterbalanced order. The primary screen remains failed.

Per-tree, per-arm values are in [`WEIGHTED_LOOKAHEAD_FORECAST_DIAGNOSTIC_20261004.csv`](WEIGHTED_LOOKAHEAD_FORECAST_DIAGNOSTIC_20261004.csv).
