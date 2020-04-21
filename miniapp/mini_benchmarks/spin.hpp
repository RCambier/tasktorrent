#include "util.hpp"

void spin_for_seconds(double time) {
    auto t0 = ttor::wctime();
    while(true) {
    	auto t1 = ttor::wctime();
        if( ttor::elapsed(t0, t1) >= time ) break;
    }
}
