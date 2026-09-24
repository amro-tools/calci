#include <calci/backend.hpp>
#include <calci/defines.hpp>
#include <calci/lennard_jones.hpp>
#include <calci/neighbourlist.hpp>
#include <stdexcept>
#include <vector>

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
        const double max_move_squared = Backend::transform_reduce<double>(
            positions.rows(),
            [&]( int n ) 
            {
                return ( position_cache.value().row( n ) - positions.row( n ) ).squaredNorm();
            },
            []( double & lhs, const double & rhs ) 
            {
                 lhs = std::max( lhs, rhs ); 
            } 
        );
        // clang-format on

        // we need to rebuild if any position changed by more than half the skin depth
        const double rebuild_distance = 0.5 * rc * verlet_skin_depth;
        rebuild_neighbour_list = max_move_squared > rebuild_distance * rebuild_distance;
    }

    if( rebuild_neighbour_list )
    {
        Calci::build_neighbour_list( cutoff, box, positions, neighbour_indices, neighbour_images );

        // cache the positions after computing the neighbour lists
        position_cache = positions;
    }
}

Vectorfield LennardJones::compute_virial( const Eigen::Ref<Vectorfield> positions )
{
    // Step 1: go through neighbour list and create ghost atoms for interaction pairs, where one of the interacting
    // atoms is in a periodic image
    const auto ghost_stuff = find_ghost_atoms( rc, box, positions );

    auto wrapped_positions = std::get<0>( ghost_stuff );
    auto ghost_atoms       = std::get<1>( ghost_stuff );
    auto idx_original      = std::get<2>( ghost_stuff );

    const int n_atoms_orig  = wrapped_positions.size();
    const int n_atoms_ghost = ghost_atoms.size();
    const int n_atoms_all   = n_atoms_orig + n_atoms_ghost;

    Vectorfield positions_all = Vectorfield( n_atoms_all, 3 );
    Vectorfield forces_all    = Vectorfield( n_atoms_all, 3 );

    // We create positions_all by assigning the wrapped position to the atoms in the original box and the ghost atoms to
    // the additional atoms
    Backend::for_each( n_atoms_all, [&]( int idx ) {
        if( idx < n_atoms_orig )
        {
            positions_all.row( idx ) = wrapped_positions[idx];
        }
        else
        {
            positions_all.row( idx ) = ghost_atoms[idx - n_atoms_orig];
        }
    } );

    auto type_ids_original = this->type_ids;

    // if type_ids is not nullopt, we have to resize it in order to fit the ghost atoms as well as the original atoms
    // and then assign typeids to the ghost atoms based on the type ids os the "original" atom
    if( type_ids.has_value() )
    {
        // resizing wont change the original type ids
        type_ids->resize( n_atoms_all );

        // therefore, we only iterate over the ghost atoms and assign the types based on idx_original
        Backend::for_each( n_atoms_ghost, [&]( int idx_ghost ) {
            const int idx_all         = n_atoms_orig + idx_ghost;
            type_ids.value()[idx_all] = type_ids_original.value()[idx_original[idx_ghost]];
        } );
    }

    // Compute energy and forces in the open boundary system for every point
    const auto box_old = box;
    box.pbc            = { false, false, false };

    // Build a half neighbour list for the open system, including the periodic ghost shell.
    position_cache = std::nullopt;
    recompute_neighbour_lists( positions_all );

    // Keep exactly one representative of each pair that crosses the original cell boundary;
    // energy_and_forces() relies on every physical pair appearing only once.
    NeighbourListIndices new_neighbour_list{};
    new_neighbour_list.resize( n_atoms_all );

#pragma omp parallel for
    for( int idx_atom = 0; idx_atom < n_atoms_all; idx_atom++ )
    {
        // iterate over the original neigbhour list
        for( auto idx_neighbour : neighbour_indices[idx_atom] )
        {
            // pair is between two atoms in the local cell -> take it
            if( idx_atom < n_atoms_orig && idx_neighbour < n_atoms_orig )
            {
                new_neighbour_list[idx_atom].push_back( idx_neighbour );
            }
            else if( idx_atom < n_atoms_orig && idx_neighbour >= n_atoms_orig )
            {
                int idx_neighbour_orig = idx_original[idx_neighbour - n_atoms_orig];
                // Only push this back if this is greater than idx_atom
                if( idx_neighbour_orig > idx_atom )
                {
                    new_neighbour_list[idx_atom].push_back( idx_neighbour );
                }
            }
            else if( idx_atom >= n_atoms_orig && idx_neighbour < n_atoms_orig )
            {
                int idx_atom_orig = idx_original[idx_atom - n_atoms_orig];
                // Only push this back if this is greater than idx_atom
                if( idx_neighbour < idx_atom_orig )
                {
                    new_neighbour_list[idx_atom].push_back( idx_neighbour );
                }
            }
        }
    }

    // Overwrite the neighbour list with our new neighbour list
    neighbour_indices = new_neighbour_list;
    // Compute virial and profit
    energy_and_forces( positions_all, forces_all );

    virial_general = Backend::transform_reduce_sum<double>( n_atoms_all, [&]( int idx ) {
        const double t = positions_all.row( idx ).dot( forces_all.row( idx ) );
        return t;
    } );

    // After we are done with the energy_and_force calculation, we reset the neighbourlist (by setting position_cache to
    // nullopt), reset the box and reset the type_ids
    position_cache = std::nullopt;
    box            = box_old;
    type_ids       = type_ids_original;

    return forces_all;
} // namespace Calci

// void debug_print( int n, int m, int type_n, int type_m, double epsilon, double sigma, double R )
// {
//     std::cout << "===\nn = " << n << ", m = " << m << ", type_n = " << type_n << ", type_m = " << type_m
//               << ", R = " << R << ", epsilon = " << epsilon << ", sigma = " << sigma << "\n";
// }

double LennardJones::energy_and_forces( const Eigen::Ref<Vectorfield> positions, Eigen::Ref<Vectorfield> forces )
{
    const int n_atoms = positions.rows();
    const double cutoff2 = rc * rc;

    check_buffers( n_atoms );

    // Clear scalar accumulators and reusable per-thread force arrays. Separate
    // arrays let both atoms of a half-list pair be updated without atomics.
    Backend::for_each( n_atoms, [&]( int i ) {
        energy_buffer_atoms[i] = 0.0;
        virial_buffer_atoms[i] = 0.0;
        for( auto & force_buffer : force_buffers_by_thread )
        {
            force_buffer.row( i ) = Vector3::Zero();
        }
    } );

    // Each half-list pair is handled once; workers accumulate into private force arrays.
    Backend::for_each( n_atoms, [&]( int n ) {
        auto & local_forces = force_buffers_by_thread[Backend::get_thread_num()];
        const int n_neighbours = static_cast<int>( neighbour_indices[n].size() );

        for( int i = 0; i < n_neighbours; ++i )
        {
            const int m = neighbour_indices[n][i];
            const Vector3 r_unwrapped = positions.row( n ) - positions.row( m );
            const Vector3 r = box.pbc_wrap( r_unwrapped ).first;
            const double R2 = r.squaredNorm();
            if( R2 > cutoff2 ) continue;

            double epsilon = this->epsilon;
            double sigma = this->sigma;
            if( type_ids.has_value() )
            {
                const int type_n = type_ids.value()[n];
                const int type_m = type_ids.value()[m];
                epsilon = epsilon_matrix( type_n, type_m );
                sigma = sigma_matrix( type_n, type_m );
            }

            // Work entirely with r^2 to avoid a square root for every pair.
            const double inv_R2 = 1.0 / R2;
            const double sigma_R_2 = sigma * sigma * inv_R2;
            const double sigma_R_6 = sigma_R_2 * sigma_R_2 * sigma_R_2;
            const double Vij_unshifted = 4.0 * epsilon * sigma_R_6 * ( sigma_R_6 - 1.0 );

            // Match ASE's smooth=false convention by making V(rc) exactly zero.
            const double sigma_rc = sigma / rc;
            const double sigma_rc_2 = sigma_rc * sigma_rc;
            const double sigma_rc_6 = sigma_rc_2 * sigma_rc_2 * sigma_rc_2;
            const double Vij =
                Vij_unshifted - 4.0 * epsilon * sigma_rc_6 * ( sigma_rc_6 - 1.0 );

            const double force_factor =
                24.0 * epsilon * inv_R2 * sigma_R_6 * ( 2.0 * sigma_R_6 - 1.0 );
            const Vector3 fij = force_factor * r;

            // Newton's third law supplies the force on m from this one evaluation.
            local_forces.row( n ) += fij;
            local_forces.row( m ) -= fij;
            energy_buffer_atoms[n] += Vij;
            virial_buffer_atoms[n] += fij.dot( r );
        }
    } );

    // Reduce private forces only after pair processing, when atom updates cannot race.
    Backend::for_each( n_atoms, [&]( int i ) {
        Vector3 total_force = Vector3::Zero();
        for( const auto & force_buffer : force_buffers_by_thread )
        {
            total_force += force_buffer.row( i );
        }
        forces.row( i ) = total_force;
    } );

    virial = Backend::sum<double>( virial_buffer_atoms );
    return Backend::sum<double>( energy_buffer_atoms );
}
} // namespace Calci
