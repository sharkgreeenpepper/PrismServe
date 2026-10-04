# Latest weighted-lookahead result

The 2026-10-04 screen stopped after block A under its frozen early-stop rule. Weight 0.5 improved tree p50 by 0.94% versus flat, below the 5% threshold; block B was not launched. All three arms completed 16/16 trees and 128/128 requests with complete required telemetry. The campaign used 83.915 GPU-min and cleaned up all services. A subsequent no-GPU diagnostic found the prior leaf output-length estimate 26.4% above the held-out mean.

See [`full results and per-tree data`](results/20261004_124417_weighted_lookahead_screen/WEIGHTED_LOOKAHEAD_RESULTS.md), [`offline forecast diagnostic`](WEIGHTED_LOOKAHEAD_FORECAST_DIAGNOSTIC_20261004.md), and [`the frozen plan`](WEIGHTED_LOOKAHEAD_PLAN.md).
