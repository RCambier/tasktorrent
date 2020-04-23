#!/bin/bash
#SBATCH -o wait_comp_benchmarks.%j.out
#SBATCH -n 1
#SBATCH -c 32

python3 wait_comp_benchmarks.py
