#!/bin/bash
#SBATCH -o ttor_sparse_cholesky_%j.out

hostname
lscpu    
mpirun -n ${SLURM_NTASKS} hostname

# mpirun -n 1 ./snchol_neglapl_driver 5 2 1 1 2
# ./snchol_neglapl_driver n d nlevels nthreads verb blocksize log folder
OMP_NUM_THREADS=1 mpirun -n ${SLURM_NTASKS} ./snchol_neglapl_driver ${N_SIZE} ${D_SIZE} ${N_LEVELS} ${N_THREADS} 0 ${BLOCK_SIZE} 0 NONE
