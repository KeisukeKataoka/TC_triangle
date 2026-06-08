#!/bin/sh

#SBATCH -J px=0.10
#SBATCH -p F1cpu
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 24:00:00


ulimit -s unlimited

srun -n $SLURM_NTASKS python triangular_EE_mpi.py 

