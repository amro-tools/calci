#pragma once
#include "calci/defines.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <calci/backend.hpp>
#include <calci/utils.hpp>

namespace Calci
{
using NeighbourListIndices = std::vector<std::vector<int>>;
using NeighbourListImages  = std::vector<std::vector<std::array<int, 3>>>;

inline void build_neighbour_list(
    double cutoff, const SimulationBoxInfo & box, const Vectorfield & coordinates,
    NeighbourListIndices & neighbour_indices, NeighbourListImages & neighbour_images )
{
    const int n_atoms = coordinates.rows();
    const double cutoff2 = cutoff * cutoff;
    const auto lattice = box.get_lattice();

    Vector3 lower = Vector3::Zero();
    Vector3 upper = Vector3::Zero();
    if( n_atoms > 0 )
    {
        lower = coordinates.colwise().minCoeff();
        upper = coordinates.colwise().maxCoeff();
    }

    std::array<int, 3> n_cells{};
    std::array<double, 3> cell_width{};
    for( int d = 0; d < 3; ++d )
    {
        const double extent = box.pbc[d] ? lattice[d] : std::max( upper[d] - lower[d], cutoff );
        n_cells[d] = std::max( 1, static_cast<int>( std::floor( extent / cutoff ) ) );
        cell_width[d] = extent / n_cells[d];
    }

    const int n_total_cells = n_cells[0] * n_cells[1] * n_cells[2];
    std::vector<std::vector<int>> cells( n_total_cells );
    auto flatten = [&]( int x, int y, int z ) { return ( x * n_cells[1] + y ) * n_cells[2] + z; };

    for( int atom = 0; atom < n_atoms; ++atom )
    {
        std::array<int, 3> cell{};
        for( int d = 0; d < 3; ++d )
        {
            double coordinate = coordinates( atom, d );
            if( box.pbc[d] )
            {
                coordinate -= std::floor( coordinate / lattice[d] ) * lattice[d];
                cell[d] = std::min( n_cells[d] - 1, static_cast<int>( coordinate / cell_width[d] ) );
            }
            else
            {
                cell[d] = std::min( n_cells[d] - 1, static_cast<int>( ( coordinate - lower[d] ) / cell_width[d] ) );
            }
        }
        cells[flatten( cell[0], cell[1], cell[2] )].push_back( atom );
    }

    neighbour_indices.assign( n_atoms, {} );
    neighbour_images.assign( n_atoms, {} );

#pragma omp parallel for
    for( int n = 0; n < n_atoms; ++n )
    {
        std::array<int, 3> home{};
        for( int d = 0; d < 3; ++d )
        {
            double coordinate = coordinates( n, d );
            if( box.pbc[d] )
            {
                coordinate -= std::floor( coordinate / lattice[d] ) * lattice[d];
                home[d] = std::min( n_cells[d] - 1, static_cast<int>( coordinate / cell_width[d] ) );
            }
            else
            {
                home[d] = std::min( n_cells[d] - 1, static_cast<int>( ( coordinate - lower[d] ) / cell_width[d] ) );
            }
        }

        std::array<int, 27> candidate_cells{};
        int n_candidate_cells = 0;
        for( int dx = -1; dx <= 1; ++dx )
        for( int dy = -1; dy <= 1; ++dy )
        for( int dz = -1; dz <= 1; ++dz )
        {
            std::array<int, 3> cell{ home[0] + dx, home[1] + dy, home[2] + dz };
            bool valid = true;
            for( int d = 0; d < 3; ++d )
            {
                if( box.pbc[d] )
                    cell[d] = ( cell[d] % n_cells[d] + n_cells[d] ) % n_cells[d];
                else if( cell[d] < 0 || cell[d] >= n_cells[d] )
                    valid = false;
            }
            if( !valid ) continue;

            const int cell_id = flatten( cell[0], cell[1], cell[2] );
            bool duplicate = false;
            for( int i = 0; i < n_candidate_cells; ++i ) duplicate |= candidate_cells[i] == cell_id;
            if( !duplicate ) candidate_cells[n_candidate_cells++] = cell_id;
        }

        for( int cell_index = 0; cell_index < n_candidate_cells; ++cell_index )
        {
            for( const int m : cells[candidate_cells[cell_index]] )
            {
                if( m <= n ) continue;
                const Vector3 r_unwrapped = coordinates.row( n ) - coordinates.row( m );
                const auto [r, img] = box.pbc_wrap( r_unwrapped );
                if( r.squaredNorm() > cutoff2 ) continue;
                neighbour_indices[n].push_back( m );
                neighbour_images[n].push_back( img );
            }
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

            Vector3 r_unwrapped = coordinates.row( n ) - coordinates.row( m );
            const auto [r, img] = box.pbc_wrap( r_unwrapped );
            const double R2     = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

            if( R2 > cutoff * cutoff )
            {
                continue;
            }
            callback( n, m, { r[0], r[1], r[2] } );
        }
    } );
}

} // namespace Calci
