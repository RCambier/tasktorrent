#include <fstream>
#include <iostream>
#include <chrono>
#include <cassert>
#include <omp.h>

void spin_for_seconds(double time) {
    auto t0 = std::chrono::high_resolution_clock::now();
    while(true) {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        if( elapsed_time >= time ) break;
    }
}

/**
 * Run n_tasks that only spins for a certain amount of time
 * Compile with something like 
 * icpc -qopenmp -O3 omp_wait.cpp -o omp_wait
 * g++ -fopenmp -O3 omp_wait.cpp -o omp_wait
 */
int wait_only(const int n_tasks, const double spin_for, const int verb) {

    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
       int n_threads = omp_get_num_threads();
       int my_thread = omp_get_thread_num();
       int tasks_per_thread = (n_tasks + n_threads - 1)/n_threads;
       int my_tasks = std::min(tasks_per_thread, n_tasks - my_thread * tasks_per_thread);
       for(int i = 0; i < my_tasks; i++) {
            #pragma omp task
            {
                spin_for_seconds(spin_for);
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    if(verb) printf("test n_threads n_taks spin_time time total_tasks efficiency\n");
    int total_tasks = n_tasks;
    int n_threads = omp_get_max_threads();
    double speedup = (double)(total_tasks) * (double)(spin_for) / (double)(time);
    double efficiency = speedup / (double)(n_threads);
    printf("omp %d %d %e %e %d %e\n", n_threads, n_tasks, spin_for, time, total_tasks, efficiency);

    return 0;
}

int main(int argc, char **argv)
{
    int n_tasks = 1000000;
    double spin_for = 1e-6;
    int verb = 0;

    if (argc >= 2)
    {
        n_tasks = atoi(argv[1]);
        assert(n_tasks >= 0);
    }
    if (argc >= 3)
    {
        spin_for = atof(argv[2]);
        assert(spin_for >= 0);
    }
    if (argc >= 4)
    {
        verb = atoi(argv[3]);
        assert(verb >= 0);
    }

    if(verb) printf("OMP_NUM_THREADS=16 ./omp_wait n_tasks spin_for verb\n");
    int error = wait_only(n_tasks, spin_for, verb);

    return error;
}
