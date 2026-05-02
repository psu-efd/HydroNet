#!/usr/bin/env bash
# Wait for the kickoff (DB_1, BP_1) to land both rows in experiment_log.csv,
# then run the remaining 13 experiments sequentially.
set -uo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
LOG="$REPO/plan/experiment_log.csv"
MAX_WAIT_S=$((90 * 60))    # 90 min ceiling on the kickoff
WAITED=0

echo "[$(date -u +%H:%M:%SZ)] Watcher armed. Waiting for DB_1 and BP_1 in $LOG ..."
while (( WAITED < MAX_WAIT_S )); do
  if [[ -f "$LOG" ]] \
     && grep -q '^DB_1,' "$LOG" 2>/dev/null \
     && grep -q '^BP_1,' "$LOG" 2>/dev/null; then
    echo "[$(date -u +%H:%M:%SZ)] Kickoff complete (waited ${WAITED}s)."
    break
  fi
  sleep 30
  WAITED=$((WAITED + 30))
done

if (( WAITED >= MAX_WAIT_S )); then
  echo "[$(date -u +%H:%M:%SZ)] WARNING: kickoff didn't complete in 90 min — running remaining matrix anyway." >&2
fi

# Grace period so the OS releases file handles before the next launch.
sleep 5

echo "[$(date -u +%H:%M:%SZ)] Launching remaining matrix: BIC_A..H + SR_A,B,C,E,F"
cd "$REPO"
bash scripts/run_all_experiments.sh \
    BIC_A BIC_B BIC_C BIC_D BIC_E BIC_F BIC_G BIC_H \
    SR_A  SR_B  SR_C  SR_E  SR_F
RC=$?
echo "[$(date -u +%H:%M:%SZ)] Matrix dispatched. Final summary -> $LOG (orchestrator exit=$RC)."
