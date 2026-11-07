#!/bin/bash
#SBATCH --job-name=fvm_scal
#SBATCH --account=xzl123_cr_usgs
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --mail-user=aletensan@gmail.com
#SBATCH --array=0-6
# R2-6 memory / step-time benchmark. Submit from examples/FVM_PINN:  sbatch scalability/bench.sbatch
TAGS=(n1306 n5000 n10000 n20000 n50000 n100000 n200000)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
module purge
cd "$SLURM_SUBMIT_DIR/savannah_river"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/storage/work/axt5780/Projects/Hydro/HydroNet/pyHMT2D
PY=/storage/work/axt5780/Projects/Hydro/HydroNet/.venv/bin/python
echo "Task $SLURM_ARRAY_TASK_ID -> $TAG on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "CPU: $(lscpu | sed -n 's/^Model name: *//p'), $SLURM_CPUS_PER_TASK cores"
if [[ $TAG == n1306 ]]; then MESH=(); else MESH=(--srhhydro data_refined/$TAG/savana_$TAG.srhhydro); fi
"$PY" ../scalability/bench_memory_steptime.py --tag "$TAG" "${MESH[@]}"
status=$?
echo "Finished $TAG at $(date), exit $status"
exit $status
