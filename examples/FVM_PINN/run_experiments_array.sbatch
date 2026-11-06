#!/bin/bash
# Slurm job array: one FVM-PINN experiment per array task, run through
# run_all_experiments.sh so per-run logs (runs/<RUN_ID>/stdout.log) and
# plan/experiment_log.csv are written exactly as in a sequential run.
#
# Task index -> RUN_ID (same order as RUN_SPEC in run_all_experiments.sh):
#   1 DB_1   2 BP_1   3 BIC_A  4 BIC_B  5 BIC_C  6 BIC_D  7 BIC_E  8 BIC_F
#   9 BIC_G 10 BIC_H 11 SR_A  12 SR_B  13 SR_C  14 SR_D  15 SR_E  16 SR_F  17 SR_G
#  18 BIC_A_GN  19 BIC_A_SA  20 BIC_A_TM  21 BIC_A_SAS   (adaptive-weighting tests, R3-C5)
#  22 BICU_A  23 BICU_B  24 BICU_E  25 BICU_H   (unsteady BIC; needs the SRH-2D
#     result data/hydrograph/block_in_channel_hydrograph_XMDFC.h5 first)
#  26 BICU_ET 27 BICU_HT 28 BICU_EW 29 BICU_HW 30 BICU_C 31 BICU_D   (unsteady follow-ups)
#  32 BICU_EWL1  33 BICU_EWL2   (BICU-EW with lambda_fvm 0.1 / 0.01)
#  34 SR_A_SF  35 SR_F_SF  36 SR_B_SF   (strong-form PINN baselines, R3-B2)
#  37 BICU_T  38 BICU_CT  39 BICU_PT  40 BICU_GT   (38-40 need 37 first: --dependency)
#  41 BIC_A_SS  42 BIC_B_SS   (steady-form residual, no IC loss; R1-M4(b))
#  43 BICU_C_NT5  44 BICU_C_NT20  45 BICU_C_NT40   (n_time_samples sweep on BICU-C; R1-M5)
#  46-60 DB_NT{1,2,5,10,20}_S{1,2,3}  61-69 DB_FX{2,5,20}_S{1,2,3}
#     (dam-break time-sampling sweep, redrawn and fixed levels, CPU; R1-M5)
#  70 BIC_B_AW  71 SR_B_AW   (FVM residual weighted by A_i/mean(A); side test)
#  72 BIC_B_CTL  73 SR_B_CTL   (unweighted controls for 70-71, on the same code)
#  74 SR_B_FL 75 SR_C_FL 76 SR_D_FL 77 SR_E_FL 78 SR_F_FL 79 SR_G_FL 80 SR_B_SF_FL 81 SR_F_SF_FL 82 SR_B_AW_FL
#     (Case 4 retrained on the flat-start SRH-2D reference)
#
# Submit from examples/FVM_PINN/:
#   GPU runs (configs with device "cuda"):
#     sbatch --array=3-5,7-17 run_experiments_array.sbatch
#   Adaptive-weighting runs (R3-C5):
#     sbatch --array=18-21 run_experiments_array.sbatch
#   Unsteady BIC (R1-M3, R2-5):
#     sbatch --array=22-25 run_experiments_array.sbatch
#     sbatch --array=26-31 run_experiments_array.sbatch   (follow-ups)
#     sbatch --array=32-33 run_experiments_array.sbatch   (lambda_fvm sweep)
#     sbatch --array=34-36 run_experiments_array.sbatch   (strong-form baselines)
#   CPU runs (configs with device "cpu": DB_1, BP_1, BIC_D):
#     sbatch --array=1,2,6 --partition=basic --gres=none --mem=32G --time=24:00:00 \
#            run_experiments_array.sbatch
#SBATCH --job-name=fvm_pinn
#SBATCH --account=xzl123_cr_usgs
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# a100 (CC 8.0) has full-rate float64; p100/v100 are unsupported by the venv's torch.
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%A_%a.out
# Summary email per run comes from job_email_report below; Slurm's own mail is
# only a backup for kills the script can't report (timeout, OOM, node failure).
#SBATCH --mail-type=FAIL,TIME_LIMIT_90,ARRAY_TASKS
#SBATCH --mail-user=aletensan@gmail.com

RUN_IDS=(_ DB_1 BP_1 BIC_A BIC_B BIC_C BIC_D BIC_E BIC_F BIC_G BIC_H
         SR_A SR_B SR_C SR_D SR_E SR_F SR_G
         BIC_A_GN BIC_A_SA BIC_A_TM BIC_A_SAS
         BICU_A BICU_B BICU_E BICU_H
         BICU_ET BICU_HT BICU_EW BICU_HW BICU_C BICU_D
         BICU_EWL1 BICU_EWL2
         SR_A_SF SR_F_SF SR_B_SF
         BICU_T BICU_CT BICU_PT BICU_GT
         BIC_A_SS BIC_B_SS
         BICU_C_NT5 BICU_C_NT20 BICU_C_NT40
         DB_NT1_S1 DB_NT1_S2 DB_NT1_S3 DB_NT2_S1 DB_NT2_S2 DB_NT2_S3 DB_NT5_S1 DB_NT5_S2 DB_NT5_S3 DB_NT10_S1 DB_NT10_S2 DB_NT10_S3 DB_NT20_S1 DB_NT20_S2 DB_NT20_S3 DB_FX2_S1 DB_FX2_S2 DB_FX2_S3 DB_FX5_S1 DB_FX5_S2 DB_FX5_S3 DB_FX20_S1 DB_FX20_S2 DB_FX20_S3
         BIC_B_AW SR_B_AW BIC_B_CTL SR_B_CTL
         SR_B_FL SR_C_FL SR_D_FL SR_E_FL SR_F_FL SR_G_FL SR_B_SF_FL SR_F_SF_FL SR_B_AW_FL)
RID=${RUN_IDS[$SLURM_ARRAY_TASK_ID]}

module purge
module load texlive/2024   # matplotlib usetex=True in the loss plots needs `latex` on PATH
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHON_BIN=/storage/work/axt5780/Projects/Hydro/HydroNet/.venv/bin/python

echo "Task $SLURM_ARRAY_TASK_ID -> $RID on $(hostname) at $(date)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv
# Hardware/software record for reporting wall times
echo "CPU: $(lscpu | sed -n 's/^Model name: *//p'), $SLURM_CPUS_PER_TASK cores allocated"
"$PYTHON_BIN" -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')"

bash run_all_experiments.sh "$RID"

# run_all_experiments.sh always exits 0; take the run's own exit code from the CSV
status=$(grep "^$RID," ../../plan/experiment_log.csv | tail -1 | cut -d, -f5)
echo "Finished $RID at $(date) with exit code ${status:-unknown}"

case_dir=$(grep -E "^\s+\"$RID " run_all_experiments.sh | awk '{print $2}')
config=$(grep -E "^\s+\"$RID " run_all_experiments.sh | awk '{print $4}' | tr -d '"')
[[ "${SKIP_EMAIL_REPORT:-0}" == 1 ]] || $HOME/bin/job_email_report --tag FVM-PINN --label "$RID" --status "${status:-1}" \
    --log "$SLURM_SUBMIT_DIR/$case_dir/runs/$RID/stdout.log" \
    --metrics 'L2[(=]|max=|Total wall time' \
    --info "config   : $case_dir/$config" \
    --paths "checkpoints: $SLURM_SUBMIT_DIR/$case_dir/checkpoints/; plots: $SLURM_SUBMIT_DIR/$case_dir/plots/$RID/"

exit "${status:-1}"
