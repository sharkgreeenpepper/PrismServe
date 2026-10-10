#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/bumi/git/PrismServe/.aris/runs/agent-memory-kv-p25-20261009
VARIANT="$ROOT/variants/qwen3-14b-cost-probe"
RUN="$VARIANT/paired-120-host-only"
MODEL=/home/bumi/infra/models/agent-memory-kv-p25-20261009/Qwen3-14B
INPUTS="$ROOT/data/controlled-calibration-120-inputs/paired-inputs.jsonl"
CONFIG="$ROOT/configs/blend.yaml"
PACKAGE=/home/bumi/git/PrismServe/experiments/agent_memory_kv_service_phase

if [[ -d "$RUN" ]]; then
  if find "$RUN" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Refusing to run into a non-empty output directory: $RUN" >&2
    exit 2
  fi
else
  mkdir -p "$RUN"
fi
source "$VARIANT/activate.sh"
date -Is > "$VARIANT/run-started.txt"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader -i 0,1,2,3 > "$VARIANT/gpu-before.txt"

sampler_pid=''
stop_sampler() {
  if [[ -n "$sampler_pid" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    sampler_pid=''
  fi
}
trap stop_sampler EXIT

run_arm() {
  local arm="$1"
  local name="$2"
  shift 2
  echo "START $name $(date -Is)"
  nvidia-smi --query-gpu=index,timestamp,memory.used,utilization.gpu --format=csv,noheader -i 0 -l 1 >> "$RUN/gpu0-samples.csv" &
  sampler_pid=$!
  CUDA_VISIBLE_DEVICES=0 python "$PACKAGE/supervise.py" python "$PACKAGE/profile_replay.py" \
    --mode host-only --event-dir "$RUN/$name-events" \
    --model "$MODEL" --inputs "$INPUTS" --output "$RUN/$name.jsonl" \
    --gpu 0 --arm "$arm" --policy window --ratio .1 --seed 17 \
    --max-model-len 32768 --max-output-tokens 32 "$@" \
    > "$RUN/$name.log" 2>&1
  stop_sampler
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader -i 0,1,2,3 > "$VARIANT/gpu-after-$name.txt"
  echo "COMPLETE $name $(date -Is)"
}

run_arm full full
run_arm strict_prefix strict-prefix
run_arm blend blend-window10 --lmcache-config "$CONFIG"
date -Is > "$VARIANT/run-finished.txt"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader -i 0,1,2,3 > "$VARIANT/gpu-release.txt"
echo "All Qwen3-14B paired arms completed at $(date -Is)"
