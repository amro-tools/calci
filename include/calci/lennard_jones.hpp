#pragma once
#include "calci/neighbourlist.hpp"
#include "calci/utils.hpp"
#include <calci/defines.hpp>
#include <optional>

namespace Calci
{
class LennardJones
{

private:
    std::optional<Vectorfield> position_cache = std::nullopt;

    Scalarfield energy_buffer_atoms{};

    Scalarfield virial_buffer_atoms{};

    void check_buffers( int n_atoms )
    {
        // Make sure the energy buffer always has the same size as the positions
        if( energy_buffer_atoms.size() != n_atoms )
        {
            energy_buffer_atoms.resize( n_atoms );
        }
        if( virial_buffer_atoms.size() != n_atoms )
        {
            virial_buffer_atoms.resize( n_atoms );
        }
    }

public:
    double sigma{};
    double epsilon{};
    double rc{};
    double ro{};
    double verlet_skin_depth{ 0.2 };
    double virial{};
    double virial_general{};

    SimulationBoxInfo box{};
    NeighbourListIndices neighbour_indices{};
    NeighbourListImages neighbour_images{};

    LennardJones( double sigma, double epsilon, double rc, double ro )
            : sigma( sigma ), epsilon( epsilon ), rc( rc ), ro( ro )
    {
    }

    void recompute_neighbour_lists( const Eigen::Ref<Vectorfield> positions );

    Vectorfield compute_virial( const Eigen::Ref<Vectorfield> positions );

    double energy_and_forces( const Eigen::Ref<Vectorfield> positions, Eigen::Ref<Vectorfield> forces );
};
} // namespace Calci
