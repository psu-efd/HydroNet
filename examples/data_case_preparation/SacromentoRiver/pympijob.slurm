#!/bin/bash

#SBATCH --nodes=1            #node = a computer, 2 nodes = 2 computers
#SBATCH --ntasks=4           #4 cores per node = 4 cores
#SBATCH --mem=12G            #using 12*4=48GB ram
#SBATCH --time=00:05:00      #1 hours, general format:DD-HHH:MM:SS
#SBATCH --job-name=pympi  #change this for each job to keep track
#SBATCH --partition=open     #using open allocation
#SBATCH --output=outputFile  #pick a name for this
#SBATCH --error=errorFile    #pick a name for this

##############################################################
# calculate number of available cores, for possible debugging purposes
#NCPU=$(srun --nodes=${SLURM_NNODES} bash -c 'hostname' | wc -l)
# number of available nodes
#NNODES=${SLURM_NNODES}

############## Print info ####################################
echo -----------INFO for Debugging and Troubleshooting-----------------
#echo "This job is allocated on ${NCPU} cpu(s)"
echo 'Job is running on node(s): '
# getting hostname for possible debugging and troubleshooting on faulty nodes
srun bash -c hostname
#echo ------------------------------------------------------
echo "SLURM: Job host is ${SLURM_SUBMIT_HOST}"
echo "SLURM: Job partition is ${SLURM_JOB_PARTITION}"
echo "SLURM: Job node list is ${SLURM_JOB_NODELIST}"
echo "SLURM: Working directory is ${SLURM_SUBMIT_DIR}"
echo "SLURM: Job id is ${SLURM_JOB_ID}"
echo "SLURM: Job name is ${SLURM_JOB_NAME}"
echo ------------------------------------------------------

cd ${SLURM_SUBMIT_DIR}
#cd ../
echo "Starting at"
date

#python run_SRH_2D_simulations.py
#mpirun -np ${NCPU} test_mpi.py
mpirun -np 4 python test_mpi.py
