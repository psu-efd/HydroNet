#!/usr/bin/env bash
# Orchestrator for the FVM-PINN SWE manuscript experiment matrix.
#
# Runs each row of the plan (plan/fvm_pinn_swe_experiment_plan.md) one at
# a time, writing logs + checkpoints into the per-case runs/ directory.
#
# Usage
# -----
#   bash examples/FVM_PINN/run_all_experiments.sh                 # run everything
#   bash examples/FVM_PINN/run_all_experiments.sh BIC_A BIC_H     # run a subset
#   DRY_RUN=1 bash examples/FVM_PINN/run_all_experiments.sh       # print commands only
#   EXTRA_ARGS="--device cuda" bash examples/FVM_PINN/run_all_experiments.sh BIC_D  # extra driver args
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
#   SR-A    : Savannah, physics only (no data of any kind) — analogue of BIC-A
#   SR-B    : Savannah, standard + SRH-2D dense
#   SR-C    : Savannah, window(5) + SRH-2D dense
#   SR-D    : Savannah, window(10) + SRH-2D dense (over-windowing)
#   SR-E    : Savannah, teacher mode — in-loop FVM trajectory + SRH-2D anchor
#   SR-F    : Savannah, standard + 200 sparse velocity measurements only
#   SR-G    : Savannah, standard, lambda_fvm = 0 + dense SRH-2D
#   BIC-A-GN: BIC-A + learning-rate annealing of loss weights (R3-C5)
#   BIC-A-SA: BIC-A + self-adaptive point weights (R3-C5)
#   BIC-A-TM: BIC-A + 10-window time-marching curriculum (R3-C5)
#   BIC-A-SAS: BIC-A-SA with SGD ascent, linear mask (R3-C5)
#   BICU-A/B/E/H: unsteady BIC (inflow pulse, t = 360-600 s): physics only /
#             10 depth gauges / gauges + dense / dense data-only (R1-M3, R2-5)
#   BICU-ET/HT: BICU-E/H with time Fourier scale 30; BICU-EW/HW: 8 time windows
#   BICU-C/D: physics + 10 fixed velocity / full-state gauges (fourier_scale_t 30)
#   BICU-EWL1/EWL2: BICU-EW with lambda_fvm = 0.1 / 0.01
#   SR-A/F/B-SF: SR-A/F/B with the strong-form PINN residual (R3-B2 baseline)
#   BICU-T: network distilled from the FVM solver only (no SRH-2D data)
#   BICU-CT/PT/GT: fine-tuned from BICU-T with physics + gauges / physics / gauges

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
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
  "SR_D   savannah_river     savannah_river_FVM_PINN.py     fvm_pinn_config_SR_D.yaml"
  "SR_E   savannah_river     savannah_river_FVM_PINN.py     fvm_pinn_config_SR_E.yaml"
  "SR_F   savannah_river     savannah_river_FVM_PINN.py     fvm_pinn_config_SR_F.yaml"
  "SR_G   savannah_river     savannah_river_FVM_PINN.py     fvm_pinn_config_SR_G.yaml"
  "BIC_A_GN block_in_channel block_in_channel_FVM_PINN.py   fvm_pinn_config_BIC_A_GN.yaml"
  "BIC_A_SA block_in_channel block_in_channel_FVM_PINN.py   fvm_pinn_config_BIC_A_SA.yaml"
  "BIC_A_TM block_in_channel block_in_channel_FVM_PINN.py   fvm_pinn_config_BIC_A_TM.yaml"
  "BIC_A_SAS block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BIC_A_SAS.yaml"
  "BICU_A   block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_A.yaml"
  "BICU_B   block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_B.yaml"
  "BICU_E   block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_E.yaml"
  "BICU_H   block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_H.yaml"
  "BICU_ET   block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_ET.yaml"
  "BICU_HT   block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_HT.yaml"
  "BICU_EW   block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_EW.yaml"
  "BICU_HW   block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_HW.yaml"
  "BICU_C    block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_C.yaml"
  "BICU_D    block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_D.yaml"
  "BICU_EWL1 block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_EWL1.yaml"
  "BICU_EWL2 block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_EWL2.yaml"
  "SR_A_SF   savannah_river     savannah_river_FVM_PINN.py     fvm_pinn_config_SR_A_SF.yaml"
  "SR_F_SF   savannah_river     savannah_river_FVM_PINN.py     fvm_pinn_config_SR_F_SF.yaml"
  "SR_B_SF   savannah_river     savannah_river_FVM_PINN.py     fvm_pinn_config_SR_B_SF.yaml"
  "BICU_T    block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_T.yaml"
  "BICU_CT   block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_CT.yaml"
  "BICU_PT   block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_PT.yaml"
  "BICU_GT   block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_GT.yaml"
  "BIC_A_SS  block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BIC_A_SS.yaml"
  "BIC_B_SS  block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BIC_B_SS.yaml"
  "BICU_C_NT5 block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_C_NT5.yaml"
  "BICU_C_NT20 block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_C_NT20.yaml"
  "BICU_C_NT40 block_in_channel block_in_channel_FVM_PINN.py  fvm_pinn_config_BICU_C_NT40.yaml"
  "BIC_B_AW block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BIC_B_AW.yaml"
  "SR_B_AW savannah_river savannah_river_FVM_PINN.py fvm_pinn_config_SR_B_AW.yaml"
  "BIC_B_CTL block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BIC_B_CTL.yaml"
  "SR_B_CTL savannah_river savannah_river_FVM_PINN.py fvm_pinn_config_SR_B_CTL.yaml"
  "SR_B_FL savannah_river savannah_river_FVM_PINN.py fvm_pinn_config_SR_B_FL.yaml"
  "SR_C_FL savannah_river savannah_river_FVM_PINN.py fvm_pinn_config_SR_C_FL.yaml"
  "SR_D_FL savannah_river savannah_river_FVM_PINN.py fvm_pinn_config_SR_D_FL.yaml"
  "SR_E_FL savannah_river savannah_river_FVM_PINN.py fvm_pinn_config_SR_E_FL.yaml"
  "SR_F_FL savannah_river savannah_river_FVM_PINN.py fvm_pinn_config_SR_F_FL.yaml"
  "SR_G_FL savannah_river savannah_river_FVM_PINN.py fvm_pinn_config_SR_G_FL.yaml"
  "SR_B_SF_FL savannah_river savannah_river_FVM_PINN.py fvm_pinn_config_SR_B_SF_FL.yaml"
  "SR_F_SF_FL savannah_river savannah_river_FVM_PINN.py fvm_pinn_config_SR_F_SF_FL.yaml"
  "SR_B_AW_FL savannah_river savannah_river_FVM_PINN.py fvm_pinn_config_SR_B_AW_FL.yaml"
  "BICU_TX450 block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BICU_TX450.yaml"
  "BICU_TX540 block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BICU_TX540.yaml"
  "BICU_CTX450 block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BICU_CTX450.yaml"
  "BICU_CTX540 block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BICU_CTX540.yaml"
  "BICU_GTX450 block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BICU_GTX450.yaml"
  "BICU_GTX540 block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BICU_GTX540.yaml"
  "DB_NT1_S1 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_NT1_S1.yaml"
  "DB_NT1_S2 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_NT1_S2.yaml"
  "DB_NT1_S3 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_NT1_S3.yaml"
  "DB_NT2_S1 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_NT2_S1.yaml"
  "DB_NT2_S2 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_NT2_S2.yaml"
  "DB_NT2_S3 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_NT2_S3.yaml"
  "DB_NT5_S1 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_NT5_S1.yaml"
  "DB_NT5_S2 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_NT5_S2.yaml"
  "DB_NT5_S3 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_NT5_S3.yaml"
  "DB_NT10_S1 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_NT10_S1.yaml"
  "DB_NT10_S2 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_NT10_S2.yaml"
  "DB_NT10_S3 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_NT10_S3.yaml"
  "DB_NT20_S1 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_NT20_S1.yaml"
  "DB_NT20_S2 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_NT20_S2.yaml"
  "DB_NT20_S3 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_NT20_S3.yaml"
  "DB_FX2_S1 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_FX2_S1.yaml"
  "DB_FX2_S2 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_FX2_S2.yaml"
  "DB_FX2_S3 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_FX2_S3.yaml"
  "DB_FX5_S1 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_FX5_S1.yaml"
  "DB_FX5_S2 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_FX5_S2.yaml"
  "DB_FX5_S3 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_FX5_S3.yaml"
  "DB_FX20_S1 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_FX20_S1.yaml"
  "DB_FX20_S2 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_FX20_S2.yaml"
  "DB_FX20_S3 dam_break_1d dam_break_1d_FVM_PINN.py nt_sweep/DB_FX20_S3.yaml"
  "BP_T channel_with_bump channel_with_bump_FVM_PINN.py fvm_pinn_config_BP_T.yaml"
  "BP_A channel_with_bump channel_with_bump_FVM_PINN.py fvm_pinn_config_BP_A.yaml"
  "BP_T_SF channel_with_bump channel_with_bump_FVM_PINN.py fvm_pinn_config_BP_T_SF.yaml"
  "BP_A_SF channel_with_bump channel_with_bump_FVM_PINN.py fvm_pinn_config_BP_A_SF.yaml"
  "BP_T_EF channel_with_bump channel_with_bump_FVM_PINN.py fvm_pinn_config_BP_T_EF.yaml"
  "BP_A_EF channel_with_bump channel_with_bump_FVM_PINN.py fvm_pinn_config_BP_A_EF.yaml"
  "DB_1_SF dam_break_1d dam_break_1d_FVM_PINN.py fvm_pinn_config_DB_1_SF.yaml"
  "BIC_A_SF block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BIC_A_SF.yaml"
  "BIC_B_SF block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BIC_B_SF.yaml"
  "BIC_E_SF block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BIC_E_SF.yaml"
  "BIC_B_S2 block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BIC_B_S2.yaml"
  "BIC_B_S3 block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BIC_B_S3.yaml"
  "BIC_G_S2 block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BIC_G_S2.yaml"
  "BIC_G_S3 block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BIC_G_S3.yaml"
  "BIC_E_S2 block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BIC_E_S2.yaml"
  "BIC_E_S3 block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BIC_E_S3.yaml"
  "BIC_H_S2 block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BIC_H_S2.yaml"
  "BIC_H_S3 block_in_channel block_in_channel_FVM_PINN.py fvm_pinn_config_BIC_H_S3.yaml"
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

  # A resumed run keeps the earlier leg's log instead of overwriting it
  if [[ "${EXTRA_ARGS:-}" == *--resume* && -f "$LOG_FILE" ]]; then
    mv "$LOG_FILE" "$RUN_DIR/stdout_$(date -r "$LOG_FILE" +%Y%m%d_%H%M%S).log"
  fi

  # shellcheck disable=SC2206
  CMD=("$PYTHON_BIN" "$DRIVER" --config "$CONFIG" --run-id "$RID" ${EXTRA_ARGS:-})

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
