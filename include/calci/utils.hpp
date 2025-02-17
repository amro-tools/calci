#pragma once
#include <array>
#include <calci/defines.hpp>
#include <vector>

namespace Calci
{
inline Vector3 mic( const Vector3 & dr, const Vector3 & cell_lengths, const std::array<bool, 3> & pbc )
{
    Vector3 wrapped_dr = dr;
    // Use minimum image convention
    for( int k = 0; k < 3; ++k )
    {
        if( pbc[k] )
        {
            wrapped_dr( k ) -= cell_lengths( k ) * std::round( dr( k ) / cell_lengths( k ) );
        }
    } // end of getting the relative distance

    return wrapped_dr;
}
} // namespace Calci