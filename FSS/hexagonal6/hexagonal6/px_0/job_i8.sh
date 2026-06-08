#!/bin/sh

#SBATCH -J h6
#SBATCH -p i8cpu
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 0:30:00


ulimit -s unlimited

srun -n $SLURM_NTASKS python triangular_EE_h6.py
