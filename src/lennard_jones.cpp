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
        Calci::build_neighbour_list_naive( cutoff, box, positions, neighbour_indices, neighbour_images );

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
    auto type_ids_all         = std::vector<int>( n_atoms_all, 0 );
    auto type_ids_original    = std::vector<int>{};

    // Create type_ids initialized to zero if the user hasn't entered anything
    if( type_ids.has_value() )
    {
        type_ids_original = type_ids.value();
    }
    else
    {
        type_ids_original = std::vector<int>( n_atoms_orig, 0 );
    }

    Backend::for_each(
        n_atoms_all,
        [&]( int idx )
        {
        if( idx < n_atoms_orig )
        {
            positions_all.row( idx ) = wrapped_positions[idx];
        }
        else
        {
            positions_all.row( idx ) = ghost_atoms[idx - n_atoms_orig];
            // Add the type ID for the ghost atom using the original index
            type_ids_all[idx] = type_ids_original[idx_original[idx - n_atoms_orig]];
        }
    } );

    // Compute energy and forces in the open boundary system for every point
    const auto box_old = box;
    box.pbc            = { false, false, false };

    // Compute full neighbour list for the system with open boundary conditions and ghost atoms in a shell of rc
    position_cache = std::nullopt;
    recompute_neighbour_lists( positions_all );

    // Generate a new full neighbour list which removes double counting for pairs straddling the boundaries of the local cell
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
    // Overwrite the type ids
    if( type_ids.has_value() )
    {
        type_ids = type_ids_all;
    }

    // Compute virial and profit
    energy_and_forces( positions_all, forces_all );

    virial_general = Backend::transform_reduce_sum<double>(
        n_atoms_all,
        [&]( int idx )
        {
        const double t = positions_all.row( idx ).dot( forces_all.row( idx ) );
        return t;
    } );

    position_cache = std::nullopt;
    box            = box_old;
    // Reset to old type_ids
    if( type_ids.has_value() )
    {
        type_ids = type_ids_original;
    }

    return forces_all;
} // namespace Calci

void debug_print(int n, int m, int type_n, int type_m, double epsilon, double sigma, double R)
{
    std::cout << "===\nn = " << n << ", m = " << m << ", type_n = " << type_n << ", type_m = " << type_m << ", R = " << R << ", epsilon = " << epsilon << ", sigma = " << sigma << "\n";
}

double LennardJones::energy_and_forces( const Eigen::Ref<Vectorfield> positions, Eigen::Ref<Vectorfield> forces )
{
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
        const double R  = r.norm();
        const double R2 = R * R;

        // fetch the "default" sigma and epsilon
        double epsilon = this->epsilon;
        double sigma   = this->sigma;

        if( type_ids.has_value() && parameter_map.has_value() )
        {
            const int type_n = type_ids.value()[n];
            const int type_m = type_ids.value()[m];
            // if the interaction is contained in the parameter map, overwrite sigma and epsilon by the value in the
            // parameter map
            if( parameter_map->contains( { type_n, type_m } ) )
            {
                epsilon = parameter_map.value()[{ type_n, type_m }].first;
                sigma   = parameter_map.value()[{ type_n, type_m }].second;
            }
        }

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

        // Divide by 2 to remove double counting of pairs (same reason we divide the energy by 2)
        virial_buffer_atoms[n] += 0.5 * fij.dot( r );
    } );

    virial = Backend::sum<double>( virial_buffer_atoms );
    return Backend::sum<double>( energy_buffer_atoms );
}
} // namespace Calci