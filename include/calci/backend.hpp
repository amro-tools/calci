#pragma once
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Calci::Backend
{

inline void set_num_threads( int n_threads )
{
#ifdef _OPENMP
    omp_set_num_threads( n_threads );
#endif
}

inline int get_num_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#endif
    return 1;
}

inline int get_thread_num()
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

template<typename CallbackT>
void for_each( int n_elements, const CallbackT & callback )
{
#pragma omp parallel for
    for( int m = 0; m < n_elements; m++ )
    {
        callback( m );
    }
}

// We define the reduction ourselves here (instead of declaring a custom omp reduction), because
// we don't want to have to declare a new reduction for each template instantiation and this
// approach works seamlessly with templates
template<typename result_t, typename CallbackT, typename ReductionT>
result_t transform_reduce( size_t n_max, const CallbackT & cb, const ReductionT & red )
{
    result_t result{ 0.0 };

#pragma omp parallel
    {
        // Each thread has its own private sum
        result_t private_result{ 0.0 };

// Distribute the loop iterations among threads
#pragma omp for nowait
        for( size_t m = 0; m < n_max; m++ )
        {
            red( private_result, cb( m ) );
        }

// Combine the private sums into the global sum
#pragma omp critical
        {
            red( result, private_result );
        }
    }

    return result;
}

// We define the reduction ourselves here (instead of declaring a custom omp reduction), because
// we don't want to have to declare a new reduction for each template instantiation and this
// approach works seamlessly with templates
template<typename result_t, typename CallbackT>
result_t transform_reduce_sum( size_t n_max, const CallbackT & cb )
{
    return transform_reduce<result_t>( n_max, cb, []( result_t & lhs, const result_t & rhs ) { lhs += rhs; } );
}

template<typename result_t, typename FieldT>
result_t sum( const FieldT & field )
{
    return transform_reduce_sum<result_t>( field.size(), [&field]( int i ) -> result_t { return field[i]; } );
}

} // namespace Calci::Backend