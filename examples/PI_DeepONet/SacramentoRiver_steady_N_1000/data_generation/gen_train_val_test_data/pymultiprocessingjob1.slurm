#!/usr/bin/env bash

#SBATCH --nodes=1            #node = a computer, 2 nodes = 2 computers
#####SBATCH --ntasks=4           #4 cores per node = 4 cores
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G     #using 12*4=48GB ram
#SBATCH --time=3:00:00      #1 hours, general format:DD-HHH:MM:SS
#SBATCH --job-name=pymulti  #change this for each job to keep track
#SBATCH --account=xzl123_cr_usgs
#SBATCH --partition=basic     #using open allocation
#SBATCH --output=outputFile1  #pick a name for this
#SBATCH --error=errorFile1    #pick a name for this

##############################################################
# Get the number of CPUs allocated to this job
NCPU=${SLURM_CPUS_ON_NODE:-2}  # Default to 2 if not set
# number of available nodes
NNODES=${SLURM_NNODES}

############## Print info ####################################
echo -----------INFO for Debugging and Troubleshooting-----------------
echo "This job is allocated ${NCPU} cpu(s) on ${NNODES} node(s)"
echo 'Job is running on node(s): '
# getting hostname for possible debugging and troubleshooting on faulty nodes
#srun bash -c hostname
#echo ------------------------------------------------------
echo "SLURM: Job host is ${SLURM_SUBMIT_HOST}"
echo "SLURM: Job partition is ${SLURM_JOB_PARTITION}"
echo "SLURM: Job node list is ${SLURM_JOB_NODELIST}"
echo "SLURM: Working directory is ${SLURM_SUBMIT_DIR}"
echo "SLURM: Job id is ${SLURM_JOB_ID}"
echo "SLURM: Job name is ${SLURM_JOB_NAME}"
echo "SLURM: CPUs per node is ${SLURM_CPUS_ON_NODE}"
echo "SLURM: Total tasks is ${SLURM_NTASKS}"
echo ------------------------------------------------------

cd ${SLURM_SUBMIT_DIR}
#cd ../
echo "Starting at"
date

module purge
module load openmpi/4.1.1-pmi2

#activate the python environment
source /storage/home/xzl123/work/python_env/HydroNet/bin/activate

echo "Bash: $(which bash)"
echo "Python path: $(which python)"
echo "MPI path: $(which mpirun)"

#python run_SRH_2D_simulations.py
#mpirun -np ${NCPU} python test_mpi.py
#mpirun -np ${NCPU} python run_SRH_2D_simulations.py
#mpirun -np ${SLURM_NTASKS} python run_SRH_2D_simulations1.py

python run_SRH_2D_simulations1.py


#srun python test_mpi.py  #does not work
#srun /storage/work/xzl123/python_env/HydroNet/bin/python test_mpi.py #runs but with cuda-related error message

echo "Finished at"
date


