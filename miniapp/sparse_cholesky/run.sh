#!/bin/bash
#SBATCH -o ttor_sparse_cholesky_scalings_%j.out

module restore spaND
hostname
lscpu    
mpirun -n ${SLURM_NTASKS} hostname

# mpirun -n 1 ./snchol_neglapl_driver 5 2 1 1 2
# ./snchol_neglapl_driver n d nlevels nthreads verb blocksize log folder
# mpirun -l -n ${SLURM_NTASKS} gdb -batch -ex=r -ex=backtrace --args ./snchol_neglapl_driver ${N_SIZE} ${D_SIZE} ${N_LEVELS} ${N_THREADS} 0 ${BLOCK_SIZE} 0 NONE 0
mpirun -l -n ${SLURM_NTASKS} ./snchol_neglapl_driver ${N_SIZE} ${D_SIZE} ${N_LEVELS} ${N_THREADS} 0 ${BLOCK_SIZE} 0 NONE 0
