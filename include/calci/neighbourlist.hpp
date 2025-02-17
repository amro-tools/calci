#pragma once
#include "calci/defines.hpp"
#include <vector>

#include <calci/backend.hpp>
#include <calci/utils.hpp>

namespace Calci
{
using NeighbourListIndices = std::vector<std::vector<int>>;

inline void build_neighbour_list_naive(
    double cutoff, const SimulationBoxInfo & box, const Vectorfield & coordinates,
    NeighbourListIndices & neighbour_indices )
{

    const int n_atoms = coordinates.rows();

    neighbour_indices.resize( n_atoms );

    const double cutoff2 = cutoff * cutoff;

#pragma omp parallel for
    for( int n = 0; n < n_atoms; n++ )
    {
        neighbour_indices[n].clear();

        // Loop through all water molecules again.
        for( int m = 0; m < n_atoms; m++ )
        {
            // Skip self-interaction in the original cell.
            if( n == m )
            {
                continue;
            }

            // Calculate coordinate difference
            Vector3 r       = coordinates.row( n ) - coordinates.row( m );
            r               = box.pbc_wrap( r );
            const double R2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

            // Skip if distance greater than cutoff
            if( R2 > cutoff2 )
            {
                continue;
            }

            neighbour_indices[n].push_back( m );
        }
    }
}

/**
 * @brief Applies `callback` to all combinations of n, m and the connecting vector r=(r_n - r_m)
 * in the neighbour list. The connecting vector r always points from the `m-site` to the
 * `n-site`.
 * For each pair, the callback is invoked as `callback(n, m, r)`, where n and m are int
 * and r is a Tensor<double,3>
 *
 * @tparam CallbackT
 * @param callback the callback function to apply
 */
template<typename CallbackT>
void iterate_neighbours(
    double cutoff, const SimulationBoxInfo & box, const Eigen::Ref<Vectorfield> coordinates,
    const NeighbourListIndices & neighbour_indices, const CallbackT & callback )
{
    const int n_atoms = coordinates.rows();
    Backend::for_each( n_atoms, [&]( int n ) {
        const int n_neighbours = neighbour_indices[n].size();

        // Iterate over all neighbours
        for( int i = 0; i < n_neighbours; i++ )
        {
            // Calculate coordinate difference
            const auto & m = neighbour_indices[n][i];

            Vector3 r       = coordinates.row( n ) - coordinates.row( m );
            r               = box.pbc_wrap( r );
            const double R2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

            if( R2 > cutoff * cutoff )
            {
                continue;
            }
            callback( n, m, { r[0], r[1], r[2] } );
        }
    } );
}

} // namespace Calci