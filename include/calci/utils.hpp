#pragma once
#include <array>
#include <calci/defines.hpp>
#include <iostream>
#include <vector>

namespace Calci
{

class SimulationBoxInfo
{
    /** size of the rectilinear simulation box. Unit: Bohr */
    std::array<double, 3> lattice{};
    std::array<double, 3> inv_lattice{};

public:
    std::array<bool, 3> pbc{};

    SimulationBoxInfo() = default;

    SimulationBoxInfo( const std::array<double, 3> & lattice, const std::array<bool, 3> & pbc )
            : lattice( lattice ), pbc( pbc )
    {
        inv_lattice = { 1.0 / lattice[0], 1.0 / lattice[1], 1.0 / lattice[2] };
    }

    void set_lattice( const std::array<double, 3> & lattice )
    {
        this->lattice = lattice;
        inv_lattice   = { 1.0 / lattice[0], 1.0 / lattice[1], 1.0 / lattice[2] };
    }

    const std::array<double, 3> & get_lattice() const
    {
        return lattice;
    }
    const std::array<double, 3> & get_inv_lattice() const
    {
        return inv_lattice;
    }

    inline std::pair<Vector3, std::array<int, 3>>
    pbc_wrap( const Vector3 & dr, const std::array<int, 3> & shift = { 0, 0, 0 } ) const
    {
        Vector3 res = dr;
        std::array<int, 3> img{};
        // Translate a coordinate by cell length if outside the box.
        for( int i = 0; i < 3; i++ )
        {
            if( pbc[i] )
            {
                res[i] = dr[i];
                img[i] = nearbyint( res[i] * inv_lattice[i] );
                res[i] -= img[i] * lattice[i];
                res[i] += static_cast<double>( shift[i] ) * lattice[i];
            }
        }
        return { res, img };
    }
};

inline Vector3 wrap_into_box( const Eigen::Ref<const Vector3> unwrapped_pos, const SimulationBoxInfo & box )
{
    Vector3 wrapped_pos    = unwrapped_pos; // initially assign the unwrapped pos
    const auto lattice     = box.get_lattice();
    const auto inv_lattice = box.get_inv_lattice();

    for( size_t i = 0; i < 3; i++ )
    {
        if( box.pbc[i] ) // only wrap if pbc are active
        {
            double fractional_coordinate = inv_lattice[i] * unwrapped_pos[i];
            fractional_coordinate -= std::floor( fractional_coordinate );
            wrapped_pos( i ) = lattice[i] * fractional_coordinate;
        }
    }
    return wrapped_pos;
}

inline std::tuple<std::vector<Vector3>, std::vector<Vector3>, std::vector<int>>
find_ghost_atoms( const double rc, const SimulationBoxInfo & box, const Eigen::Ref<Vectorfield> positions )
{
    // iterate over all atoms
    //  ... fold them back into the box
    //  ... check how far each atom is from the box boundary
    //  ... if it is within rc of the box boundary, create a ghost atom
    const int n_atoms                   = positions.rows();
    const std::array<double, 3> lattice = box.get_lattice();

    std::vector<Vector3> wrapped_positions{};
    std::vector<Vector3> ghost_atoms{};
    std::vector<int> idx_original{};

    // the number of periodic images to test to either side in x/y/z direction
    // if no periodic boundary conditions are active the number is zero, else it is one
    const int nx = box.pbc[0] ? 1 : 0;
    const int ny = box.pbc[1] ? 1 : 0;
    const int nz = box.pbc[2] ? 1 : 0;

#pragma omp parallel
    {
        // To avoid race conditions (because of std::vector::push_back) we use a private wrapped_positions vector and a
        std::vector<Vector3> wrapped_positions_thread{};

        std::vector<Vector3> ghost_atoms_thread;
        std::vector<int> idx_original_thread{};

#pragma omp for nowait
        for( int i = 0; i < n_atoms; i++ )
        {
            // 1. fold back
            const Vector3 unwrapped_pos = positions.row( i );
            const Vector3 p             = wrap_into_box( unwrapped_pos, box );

            wrapped_positions_thread.push_back( p );

            // We have to check the surrounding eight (upto) periodic images
            // ... we address each periodic image by the translations in x, y and z direction
            for( int ix = -nx; ix <= nx; ix++ )
            {
                for( int iy = -ny; iy <= ny; iy++ )
                {
                    for( int iz = -nz; iz <= nz; iz++ )
                    {
                        // skip the original cell
                        if( ( ix == 0 ) && ( iy == 0 ) && ( iz == 0 ) )
                        {
                            continue;
                        }

                        // compute the position of the atom in the periodic image by translating with the lattice vectors
                        const Vector3 p_img
                            = { p[0] + ix * lattice[0], p[1] + iy * lattice[1], p[2] + iz * lattice[2] };

                        // then, we check if the atom in the periodic image is within a distance of `rc` of the boundary
                        // of the original cell
                        bool add_ghost_atom = true;

                        // loop over the components of p_img
                        for( int ic = 0; ic < 3; ic++ )
                        {
                            // only add the ghost atom if all components r[ic] fulfill
                            //    r[ic] \in [ -rc, L[ic] + rc ]
                            const bool check_comp = p_img[ic] > -rc && p_img[ic] < lattice[ic] + rc;
                            if( !check_comp )
                            {
                                add_ghost_atom = false;
                                break;
                            }
                        }
                        if( add_ghost_atom )
                        {
                            ghost_atoms_thread.push_back( p_img );
                            idx_original_thread.push_back( i );
                        }
                    }
                }
            }
        }

        // Combine the private wrapped_positions and ghost_atoms into the final result
#pragma omp critical
        {
            wrapped_positions.insert(
                wrapped_positions.end(), wrapped_positions_thread.begin(), wrapped_positions_thread.end() );
            ghost_atoms.insert( ghost_atoms.end(), ghost_atoms_thread.begin(), ghost_atoms_thread.end() );
            idx_original.insert( idx_original.end(), idx_original_thread.begin(), idx_original_thread.end() );
        }
    }
    // return { wrapped_positions, ghost_atoms, index_map };
    return { wrapped_positions, ghost_atoms, idx_original };
}

} // namespace Calci