#ifdef _OPENMP
#include <omp.h>
#endif

#include <calci/defines.hpp>
#include <calci/utils.hpp>
#include <chrono>
#include <format>
#include <iostream>

int get_n_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1.0;
#endif
}

int main() {}