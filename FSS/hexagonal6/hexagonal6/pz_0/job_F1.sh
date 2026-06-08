#!/bin/sh

#SBATCH -J h6_pz0
#SBATCH -p F1cpu
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 24:00:00


ulimit -s unlimited

srun -n $SLURM_NTASKS python triangular_EE_h6.py

