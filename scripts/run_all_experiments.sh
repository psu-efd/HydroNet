#!/usr/bin/env bash
# Orchestrator for the FVM-PINN SWE manuscript experiment matrix.
#
# Runs each row of the plan (plan/fvm_pinn_swe_experiment_plan.md) one at
# a time, writing logs + checkpoints into the per-case runs/ directory.
#
# Usage
# -----
#   bash scripts/run_all_experiments.sh                 # run everything
#   bash scripts/run_all_experiments.sh BIC_A BIC_H     # run a subset
#   DRY_RUN=1 bash scripts/run_all_experiments.sh       # print commands only
#
# Pre-reqs
# --------
#   - .venv activated with HydroNet installed (pip install -e .)
#   - NVIDIA GPU visible to PyTorch (RTX 4000 assumed in the plan)
#   - SRH-2D case files in examples/FVM_PINN/<case>/data/ (already in repo)
#
# Output layout
# -------------
# Each run writes into
#     examples/FVM_PINN/<case>/runs/<RUN_ID>/
#         stdout.log           (tee of the training output)
#         checkpoints/...      (the script's save dir — see YAML)
#         comparison figures   (from the example script's evaluate_and_plot)
# and appends one row to plan/experiment_log.csv with
#     run_id, config, wall_time_s, exit_code, timestamp
#
# Runs currently enabled
# ----------------------
#   DB-1    : 1D dam break, physics only (Stoker validation)         (CPU ok)
#   BP-1    : 1D bump, teacher mode (base config)
#   BIC-A   : 2D block, standard, lambda_data = 0 (physics-only fail)
#   BIC-B   : 2D block, standard + 200 sparse velocity measurements
#   BIC-C   : 2D block, standard + 50 sparse velocity measurements
#   BIC-D   : 2D block, teacher mode (base config) — dense FVM-snapshot anchor
#   BIC-E   : 2D block, standard + sparse(200) + dense SRH-2D snapshots
#   BIC-F   : 2D block, standard + 200 sparse measurements with 5% noise
#   BIC-G   : 2D block, data-only (lambda_fvm = 0) + 200 sparse measurements
#   BIC-H   : 2D block, data-only (lambda_fvm = 0) + dense SRH-2D snapshots
#   SR-A    : Savannah, standard + SRH-2D dense
#   SR-B    : Savannah, window(5) + SRH-2D dense
#   SR-C    : Savannah, window(10) + SRH-2D dense (over-windowing)
#   SR-E    : Savannah, standard + 200 sparse velocity measurements only
#   SR-F    : Savannah, standard, lambda_fvm = 0 + dense SRH-2D

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
EXAMPLES="$REPO_ROOT/examples/FVM_PINN"
LOG_CSV="$REPO_ROOT/plan/experiment_log.csv"

# Resolve the Python interpreter. Prefer the project-local .venv so that
# PATH-not-activated invocations (background runs, fresh shells) still find
# the right torch + HydroNet install. Override by exporting PYTHON_BIN.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/Scripts/python.exe"   # Windows venv
  elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"           # POSIX venv
  else
    PYTHON_BIN="python"                                 # fall back to PATH
  fi
fi
echo "Using PYTHON_BIN=$PYTHON_BIN"

# ----- run registry: RUN_ID | case_dir | driver script | config YAML -----
# Order is the recommended execution ordering from the plan.
declare -a RUN_SPEC=(
  "DB_1   dam_break_1d       dam_break_1d_FVM_PINN.py       fvm_pinn_config.yaml"
  "BP_1   channel_with_bump  channel_with_bump_FVM_PINN.py  fvm_pinn_config.yaml"
  "BIC_A  block_in_channel   block_in_channel_FVM_PINN.py   fvm_pinn_config_BIC_A.yaml"
  "BIC_B  block_in_channel   block_in_channel_FVM_PINN.py   fvm_pinn_config_BIC_B.yaml"
  "BIC_C  block_in_channel   block_in_channel_FVM_PINN.py   fvm_pinn_config_BIC_C.yaml"
  "BIC_D  block_in_channel   block_in_channel_FVM_PINN.py   fvm_pinn_config.yaml"
  "BIC_E  block_in_channel   block_in_channel_FVM_PINN.py   fvm_pinn_config_BIC_E.yaml"
  "BIC_F  block_in_channel   block_in_channel_FVM_PINN.py   fvm_pinn_config_BIC_F.yaml"
  "BIC_G  block_in_channel   block_in_channel_FVM_PINN.py   fvm_pinn_config_BIC_G.yaml"
  "BIC_H  block_in_channel   block_in_channel_FVM_PINN.py   fvm_pinn_config_BIC_H.yaml"
  "SR_A   savannah_river     savannah_river_FVM_PINN.py     fvm_pinn_config_SR_A.yaml"
  "SR_B   savannah_river     savannah_river_FVM_PINN.py     fvm_pinn_config_SR_B.yaml"
  "SR_C   savannah_river     savannah_river_FVM_PINN.py     fvm_pinn_config_SR_C.yaml"
  "SR_E   savannah_river     savannah_river_FVM_PINN.py     fvm_pinn_config_SR_E.yaml"
  "SR_F   savannah_river     savannah_river_FVM_PINN.py     fvm_pinn_config_SR_F.yaml"
)

# Parse filter args (if any)
if [[ $# -gt 0 ]]; then
  FILTER=("$@")
else
  FILTER=()
fi

want_run() {
  local rid="$1"
  if [[ ${#FILTER[@]} -eq 0 ]]; then return 0; fi
  for w in "${FILTER[@]}"; do
    [[ "$w" == "$rid" ]] && return 0
  done
  return 1
}

# Ensure CSV exists + has header
if [[ ! -f "$LOG_CSV" ]]; then
  mkdir -p "$(dirname "$LOG_CSV")"
  echo "run_id,case_dir,config,wall_time_s,exit_code,timestamp" > "$LOG_CSV"
fi

for spec in "${RUN_SPEC[@]}"; do
  # shellcheck disable=SC2206
  parts=($spec)
  RID="${parts[0]}"
  CASE_DIR="${parts[1]}"
  DRIVER="${parts[2]}"
  CONFIG="${parts[3]}"

  if ! want_run "$RID"; then continue; fi

  WORKDIR="$EXAMPLES/$CASE_DIR"
  RUN_DIR="$WORKDIR/runs/$RID"
  LOG_FILE="$RUN_DIR/stdout.log"

  echo ""
  echo "=================================================================="
  echo "[$RID] $CASE_DIR :: python $DRIVER --config $CONFIG"
  echo "  workdir : $WORKDIR"
  echo "  run_dir : $RUN_DIR"
  echo "=================================================================="

  mkdir -p "$RUN_DIR"

  CMD=("$PYTHON_BIN" "$DRIVER" --config "$CONFIG")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN: (cd $WORKDIR && ${CMD[*]}) | tee $LOG_FILE"
    continue
  fi

  START_TS="$(date +%s)"
  EXIT_CODE=0
  (cd "$WORKDIR" && "${CMD[@]}") 2>&1 | tee "$LOG_FILE" || EXIT_CODE=$?
  END_TS="$(date +%s)"
  WALL=$(( END_TS - START_TS ))

  # Append metrics row. timestamp in ISO-8601 UTC
  TS_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "$RID,$CASE_DIR,$CONFIG,$WALL,$EXIT_CODE,$TS_ISO" >> "$LOG_CSV"

  if [[ $EXIT_CODE -ne 0 ]]; then
    echo "[!!] $RID failed with exit $EXIT_CODE — continuing to next run." >&2
  fi
done

echo ""
echo "All requested runs dispatched. Summary -> $LOG_CSV"
