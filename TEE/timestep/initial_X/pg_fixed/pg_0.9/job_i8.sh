#!/bin/sh

#SBATCH -J timestep
#SBATCH -p i8cpu
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 0:30:00


ulimit -s unlimited

srun -n $SLURM_NTASKS python triangular_ranyi2_timestep_pg_fixed_v3.py
