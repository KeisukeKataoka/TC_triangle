#!/bin/sh

#SBATCH -J h1
#SBATCH -p F1cpu
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 24:00:00


ulimit -s unlimited

srun -n $SLURM_NTASKS python triangular_EE_h3_mpi.py

