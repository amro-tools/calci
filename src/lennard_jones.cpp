#include <calci/backend.hpp>
#include <calci/defines.hpp>
#include <calci/lennard_jones.hpp>
#include <calci/neighbourlist.hpp>

namespace Calci
{

void LennardJones::recompute_neighbour_lists( const Eigen::Ref<Vectorfield> positions )
{
    // Add the skin depth to the cutoff
    const double cutoff = rc * ( 1.0 + verlet_skin_depth );

    // This bool will decide if we rebuild the neighbour list
    bool rebuild_neighbour_list{};

    // If the cache does not have a value or the verlet_skin-depth is
    // zero, we always need to rebuild
    if( !position_cache.has_value() || verlet_skin_depth <= 0 )
    {
        rebuild_neighbour_list = true;
    }
    else // else we check the maxmove
    {
        // Determine the maximum distance moved since the
        // last neighbourlist update

        // clang-format off
        const double max_move = Backend::transform_reduce<double>(
            positions.size(), 
            [&]( int n ) 
            {
                return ( position_cache.value().row( n ) - positions.row( n ) ).norm(); 
            },
            []( double & lhs, const double & rhs ) 
            {
                 lhs = std::max( lhs, rhs ); 
            } 
        );
        // clang-format on

        // we need to rebuild if any position changed by more than half the skin depth
        rebuild_neighbour_list = max_move > 0.5 * rc * verlet_skin_depth;
    }

    if( rebuild_neighbour_list )
    {
        Calci::build_neighbour_list_naive( cutoff, box, positions, neighbour_indices );

        // cache the positions after computing the neighbour lists
        position_cache = positions;
    }
}

double LennardJones::energy_and_forces( const Eigen::Ref<Vectorfield> positions, Eigen::Ref<Vectorfield> forces )
{
    recompute_neighbour_lists( positions );

    const int n_atoms = positions.rows();

    check_buffers( n_atoms );

    // zero out the forces and energy buffer
    Backend::for_each(
        n_atoms,
        [&]( int i )
        {
            forces.row( i )        = Vector3::Zero();
            energy_buffer_atoms[i] = 0.0;
            virial_buffer_atoms[i] = 0.0;
        } );

    iterate_neighbours(
        rc, box, positions, neighbour_indices,
        [&]( int n, int m, const Vector3 & r )
        {
            const double R         = r.norm();
            const double R2        = R * R;
            const double sigma_R   = sigma / R;
            const double sigma_R_2 = sigma_R * sigma_R;
            const double sigma_R_4 = sigma_R_2 * sigma_R_2;
            const double sigma_R_6 = sigma_R_4 * sigma_R_2;

            // V_{ij} = 4 \epsilon * ( (\sigma/R_{ij})^12 - (\sigma/R_{ij})^6 )
            const double Vij = 4.0 * epsilon * sigma_R_6 * ( sigma_R_6 - 1.0 );

            // factor of 1/2 to account for double counting in the neighbour list
            energy_buffer_atoms[n] += 0.5 * Vij;

            const double dVij_drij = 24.0 * epsilon / R2 * sigma_R_6 * ( 2.0 * sigma_R_6 - 1.0 );

            const Vector3 fij = dVij_drij * r;
            forces.row( n ) += fij;

            virial_buffer_atoms[n] += fij.dot( r );
        } );

    virial = Backend::sum<double>( virial_buffer_atoms );
    return Backend::sum<double>( energy_buffer_atoms );
}
} // namespace Calci