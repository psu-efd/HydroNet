#!/usr/bin/env bash
# After the main matrix (bab0lstp5) finishes, re-run the three runs that
# failed against the *unpatched* var_mask shape:
#   BIC_D  (teacher trainer fix)
#   SR_B   (window trainer fix, 5 windows)
#   SR_C   (window trainer fix, 10 windows)
#
# We wait for SR_F to land in the CSV before launching, since SR_F is
# the last run in the original matrix order. (SR_F either succeeds or
# fails — either way it produces a CSV row, which is our signal.)
set -uo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
LOG="$REPO/plan/experiment_log.csv"
MAX_WAIT_S=$((12 * 60 * 60))   # 12-hour cap on the wait
WAITED=0

echo "[$(date -u +%H:%M:%SZ)] Watcher armed. Waiting for SR_F in $LOG ..."
while (( WAITED < MAX_WAIT_S )); do
  if [[ -f "$LOG" ]] && grep -q '^SR_F,' "$LOG" 2>/dev/null; then
    echo "[$(date -u +%H:%M:%SZ)] SR_F landed (waited ${WAITED}s). Launching re-runs."
    break
  fi
  sleep 60
  WAITED=$((WAITED + 60))
done

if (( WAITED >= MAX_WAIT_S )); then
  echo "[$(date -u +%H:%M:%SZ)] WARNING: SR_F never landed in 12h — re-running anyway." >&2
fi

sleep 5  # grace period

echo "[$(date -u +%H:%M:%SZ)] Re-running BIC_D + SR_B + SR_C with patched trainers."
cd "$REPO"
bash scripts/run_all_experiments.sh BIC_D SR_B SR_C
RC=$?
echo "[$(date -u +%H:%M:%SZ)] Re-runs dispatched. Final summary -> $LOG (orchestrator exit=$RC)."
