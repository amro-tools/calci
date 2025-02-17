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

public:
    double sigma{};
    double epsilon{};
    double rc{};
    double ro{};
    double verlet_skin_depth{ 0.2 };

    SimulationBoxInfo box{};
    NeighbourListIndices neighbour_indices{};

    LennardJones( double sigma, double epsilon, double rc, double ro )
            : sigma( sigma ), epsilon( epsilon ), rc( rc ), ro( ro )
    {
    }

    void recompute_neighbour_lists( const Eigen::Ref<Vectorfield> positions );

    double energy_and_forces( const Eigen::Ref<Vectorfield> positions, Eigen::Ref<Vectorfield> forces );
};
} // namespace Calci
