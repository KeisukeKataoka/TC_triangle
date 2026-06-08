#!/bin/sh

#SBATCH -J h3
#SBATCH -p i8cpu
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 0:30:00


ulimit -s unlimited

srun -n $SLURM_NTASKS python triangular_EE_h3_mpi.py

#srun -n 128 triangular_EE_h3_mpi.py