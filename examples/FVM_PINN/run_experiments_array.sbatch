#!/bin/bash
# Slurm job array: one FVM-PINN experiment per array task, run through
# run_all_experiments.sh so per-run logs (runs/<RUN_ID>/stdout.log) and
# plan/experiment_log.csv are written exactly as in a sequential run.
#
# Task index -> RUN_ID (same order as RUN_SPEC in run_all_experiments.sh):
#   1 DB_1   2 BP_1   3 BIC_A  4 BIC_B  5 BIC_C  6 BIC_D  7 BIC_E  8 BIC_F
#   9 BIC_G 10 BIC_H 11 SR_A  12 SR_B  13 SR_C  14 SR_D  15 SR_E  16 SR_F  17 SR_G
#
# Submit from examples/FVM_PINN/:
#   GPU runs (configs with device "cuda"):
#     sbatch --array=3-5,7-17 run_experiments_array.sbatch
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
         SR_A SR_B SR_C SR_D SR_E SR_F SR_G)
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
$HOME/bin/job_email_report --tag FVM-PINN --label "$RID" --status "${status:-1}" \
    --log "$SLURM_SUBMIT_DIR/$case_dir/runs/$RID/stdout.log" \
    --metrics 'L2[(=]|max=|Total wall time' \
    --info "config   : $case_dir/$config" \
    --paths "checkpoints: $SLURM_SUBMIT_DIR/$case_dir/checkpoints/; plots: $SLURM_SUBMIT_DIR/$case_dir/plots/$RID/"

exit "${status:-1}"
