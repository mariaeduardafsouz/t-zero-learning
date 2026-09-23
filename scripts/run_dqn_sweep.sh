#!/usr/bin/env bash
# DQN assignment sweep — baseline + the hyperparameter sweeps described in
# assignments/dqn.md (Part 3: Q1 target_network_frequency, Q2 buffer_size, Q3 your choice).
#
# Run this only after algorithms/dqn.py's three YOUR CODE HERE blocks pass
# `python -m pytest tests/test_dqn.py`.
#
# Usage (from repo root, with .venv activated):
#   ./scripts/run_dqn_sweep.sh
#
# Environment variables:
#   PARALLEL          Max trainings running at once (default: 3). Each run is
#                      pinned to 1 thread (OMP/MKL) to avoid CPU oversubscription.
#   TOTAL_TIMESTEPS   Override dqn_cartpole's total_timesteps for every run
#                      (default: unset = config default, 500000). Use a small
#                      value (e.g. 20000) for a dry smoke pass before committing
#                      to the full ~10-20 min/run sweep.
#   FILTER_LABELS     Comma-separated subset of labels to run (default: all —
#                      see RUNS below). Useful to re-run one config or to test
#                      the script itself, e.g. FILTER_LABELS=baseline.
#   EXTRA_OVERRIDE    Extra --override args appended to every run (space-separated),
#                      e.g. EXTRA_OVERRIDE="track=false capture_video=false" for
#                      a local dry run that doesn't hit wandb.
#   LOG_DIR           Where per-run logs go (default: logs/dqn_sweep).
#   DRY_RUN           If 1, print the commands instead of running them.
#
# Runs launched by default (12 total — matches the values already used in the
# draft report; edit RUNS below to change the sweep). q1/q2/q3 configs below
# already have a manual seed=2 run from before this script was used (exp_name
# "empty") — only seed=1 is queued for those, to fill the second-seed gap
# without re-spending compute. baseline and the large buffer are brand new
# (both seeds), since neither existed yet.
#   baseline                    (default config)     seeds 1,2  [NEW]
#   Q1  dqn.target_network_frequency=1                seed  1   [seed=2 exists]
#   Q1  dqn.target_network_frequency=100               seed  1   [seed=2 exists]
#   Q1  dqn.target_network_frequency=1000              seed  1   [seed=2 exists]
#   Q2  dqn.buffer_size=1                              seed  1   [seed=2 exists]
#   Q2  dqn.buffer_size=510                            seed  1   [seed=2 exists]
#   Q2  dqn.buffer_size=200000                         seeds 1,2 [NEW]
#   Q3  dqn.gamma=0.1                                  seed  1   [seed=2 exists]
#   Q3  dqn.gamma=0.8                                  seed  1   [seed=2 exists]
#   Q3  dqn.gamma=0.9                                  seed  1   [seed=2 exists]
#
# Each run's exp_name is set to dqn_sweep/<label>, so results land under
# runs/dqn_sweep/<label>/... . wandb run *names* do not encode the swept
# hyperparameter (see algorithms/base.py) — in the wandb UI, group or color
# charts by the relevant dqn.* config field instead.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PARALLEL="${PARALLEL:-3}"
LOG_DIR="${LOG_DIR:-logs/dqn_sweep}"
DRY_RUN="${DRY_RUN:-0}"
FILTER_LABELS="${FILTER_LABELS:-}"
mkdir -p "$LOG_DIR"

# label|override(without seed/exp_name, may be empty)|space-separated seeds
RUNS=(
  "baseline||1 2"
  "q1_tnf1|dqn.target_network_frequency=1|1"
  "q1_tnf100|dqn.target_network_frequency=100|1"
  "q1_tnf1000|dqn.target_network_frequency=1000|1"
  "q2_buf1|dqn.buffer_size=1|1"
  "q2_buf510|dqn.buffer_size=510|1"
  "q2_buf200000|dqn.buffer_size=200000|1 2"
  "q3_gamma0.1|dqn.gamma=0.1|1"
  "q3_gamma0.8|dqn.gamma=0.8|1"
  "q3_gamma0.9|dqn.gamma=0.9|1"
)

if [[ -n "${TOTAL_TIMESTEPS:-}" ]]; then
  echo "NOTE: overriding total_timesteps=$TOTAL_TIMESTEPS for every run in this sweep"
fi

extra_arr=()
if [[ -n "${EXTRA_OVERRIDE:-}" ]]; then
  read -r -a extra_arr <<< "$EXTRA_OVERRIDE"
fi

should_run() {
  local label="$1"
  [[ -z "$FILTER_LABELS" ]] && return 0
  local IFS=','
  local wanted
  for wanted in $FILTER_LABELS; do
    [[ "$wanted" == "$label" ]] && return 0
  done
  return 1
}

pids=()
fail=0
launched=0

for entry in "${RUNS[@]}"; do
  IFS='|' read -r label override seeds <<< "$entry"
  should_run "$label" || continue
  for seed in $seeds; do
    override_values=("seed=${seed}" "exp_name=dqn_sweep/${label}")
    [[ -n "$override" ]] && override_values+=("$override")
    [[ -n "${TOTAL_TIMESTEPS:-}" ]] && override_values+=("total_timesteps=${TOTAL_TIMESTEPS}")
    [[ ${#extra_arr[@]} -gt 0 ]] && override_values+=("${extra_arr[@]}")

    log="$LOG_DIR/${label}_seed${seed}.log"
    cmd=(python train.py --config dqn_cartpole --override "${override_values[@]}")

    echo "=== [$((++launched))] ${label} seed=${seed} -> ${log} ==="
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '%q ' "${cmd[@]}"; echo
      continue
    fi

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "${cmd[@]}" >"$log" 2>&1 &
    pids+=("$!")
    # Portable throttle (no `wait -n`: macOS ships bash 3.2). Exit statuses are
    # still captured correctly below — `wait "$pid"` works even for a job that
    # already finished, as long as nothing else has reaped it first.
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do
      sleep 1
    done
  done
done

for p in "${pids[@]:-}"; do
  [[ -n "$p" ]] && { wait "$p" || fail=1; }
done

if (( fail )); then
  echo "ERROR: one or more sweep runs failed. Check logs under $LOG_DIR" >&2
  exit 1
fi

echo "Done. $launched run(s) launched. Logs under $LOG_DIR"
echo "In wandb: group/color charts by the swept dqn.* config field (run names don't encode it)."
