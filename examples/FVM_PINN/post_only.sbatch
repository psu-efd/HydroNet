#!/bin/bash
# Evaluation + plotting only (--post-only) for a run whose training already
# finished and saved its final checkpoint. Writes runs/<RUN_ID>/post_only.log
# so the training log (stdout.log) is kept.
#
# Submit from examples/FVM_PINN/, one job per run:
#   sbatch --job-name=post_BIC_A post_only.sbatch BIC_A
# Optional 2nd arg: checkpoint path relative to the case dir (default: the
# driver's final checkpoint), e.g. for a run killed during L-BFGS:
#   sbatch --job-name=post_BIC_E post_only.sbatch BIC_E checkpoints/BIC_E/ckpt_adam_8000.pt
#SBATCH --account=xzl123_cr_usgs
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aletensan@gmail.com

RID=$1
CKPT=$2
spec=$(grep -E "^\s+\"$RID " run_all_experiments.sh | tr -d '"')
case_dir=$(echo "$spec" | awk '{print $2}')
driver=$(echo "$spec" | awk '{print $3}')
config=$(echo "$spec" | awk '{print $4}')
[ -z "$driver" ] && { echo "Unknown RUN_ID '$RID'"; exit 2; }

module purge
module load texlive/2024   # matplotlib usetex=True needs `latex` on PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PYTHON_BIN=/storage/work/axt5780/Projects/Hydro/HydroNet/.venv/bin/python

log="$SLURM_SUBMIT_DIR/$case_dir/runs/$RID/post_only.log"
mkdir -p "$(dirname "$log")"
echo "Post-only $RID: $case_dir/$driver --config $config on $(hostname) at $(date)"
(cd "$SLURM_SUBMIT_DIR/$case_dir" && "$PYTHON_BIN" "$driver" --config "$config" --post-only --device cpu --run-id "$RID" ${CKPT:+--checkpoint "$CKPT"}) 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
echo "Finished $RID at $(date) with exit code $status"

$HOME/bin/job_email_report --tag "FVM-PINN eval" --label "$RID" --status "$status" \
    --log "$log" --metrics 'L2[(=]|max=' \
    --info "config   : $case_dir/$config (--post-only${CKPT:+, $CKPT})" \
    --paths "plots: $SLURM_SUBMIT_DIR/$case_dir/plots/$RID/"

exit "$status"
