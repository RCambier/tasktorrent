#include "runtime.hpp"
#include "util.hpp"
#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <memory>

#include "spin.hpp"

/**
 * Run n_tasks that only spins for a certain amount of time
 */
int wait_only(const int n_threads, const int n_tasks, const double spin_for, const int verb) {

    ttor::Threadpool_shared tp(n_threads, 0, "Wk_", false);
    ttor::Taskflow<int> tf_0(&tp, 0);

    tf_0.set_mapping([&](int k) {
        return (k % n_threads);
    })
    .set_indegree([&](int k) {
        return 1;
    })
    .set_task([&](int k) {
        spin_for_seconds(spin_for);
    });


    for(int k = 0; k < n_tasks; k++) {
        tf_0.fulfill_promise(k);
    }
    auto t0 = ttor::wctime();
    tp.start();
    tp.join();
    auto t1 = ttor::wctime();
    double time = ttor::elapsed(t0, t1);
    if(verb) printf("test n_threads n_taks spin_time time total_tasks efficiency\n");
    int total_tasks = n_tasks;
    double speedup = (double)(total_tasks) * (double)(spin_for) / (double)(time);
    double efficiency = speedup / (double)(n_threads);
    printf("ttor %d %d %e %e %d %e\n", n_threads, n_tasks, spin_for, time, total_tasks, efficiency);

    return 0;
}

int main(int argc, char **argv)
{
    int n_threads = 1;
    int n_tasks = 1000000;
    double spin_for = 1e-6;
    int verb = 0;

    if (argc >= 2)
    {
        n_threads = atoi(argv[1]);
        assert(n_threads > 0);
    }
    if (argc >= 3)
    {
        n_tasks = atoi(argv[2]);
        assert(n_tasks >= 0);
    }
    if (argc >= 4)
    {
        spin_for = atof(argv[3]);
        assert(spin_for >= 0);
    }
    if (argc >= 5)
    {
        verb = atoi(argv[4]);
        assert(verb >= 0);
    }

    if(verb) printf("./micro_wait n_threads n_tasks spin_for verb\n");
    int error = wait_only(n_threads, n_tasks, spin_for, verb);

    return error;
}
