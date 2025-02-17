#pragma once
#include <array>
#include <calci/defines.hpp>

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

    inline Vector3 pbc_wrap( const Vector3 & dr, const std::array<int, 3> & shift = { 0, 0, 0 } ) const
    {
        Vector3 res = dr;
        // Translate a coordinate by cell length if outside the box.
        for( int i = 0; i < 3; i++ )
        {
            if( pbc[i] )
            {
                res[i] = dr[i];
                res[i] -= nearbyint( res[i] * inv_lattice[i] ) * lattice[i];
                res[i] += static_cast<double>( shift[i] ) * lattice[i];
            }
        }
        return res;
    }
};

} // namespace Calci