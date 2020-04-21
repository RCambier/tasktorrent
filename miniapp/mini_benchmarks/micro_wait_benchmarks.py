import subprocess
import os

for threads in [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]:
    for time in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        for repeat in range(1):
            tasks = threads * 1.0 / max(time, 1e-6)
            subprocess.run(["./ttor_wait", str(threads), str(tasks), str(time), "0"])
            os.environ['OMP_NUM_THREADS'] = str(threads)
            subprocess.run(["./omp_wait", str(tasks), str(time), "0"])
