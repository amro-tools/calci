#pragma once
#include "calci/neighbourlist.hpp"
#include "calci/utils.hpp"
#include <calci/defines.hpp>
#include <algorithm>
#include <optional>
#include <stdexcept>

namespace Calci
{
class LennardJones
{

private:
    std::optional<Vectorfield> position_cache = std::nullopt;

    Scalarfield energy_buffer_atoms{};

    Scalarfield virial_buffer_atoms{};

    std::vector<Vectorfield> force_buffers_by_thread{};

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
        const int n_threads = Backend::get_num_threads();
        force_buffers_by_thread.resize( n_threads );
        for( auto & force_buffer : force_buffers_by_thread )
        {
            if( force_buffer.rows() != n_atoms )
            {
                force_buffer.resize( n_atoms, 3 );
            }
        }
        if( type_ids.has_value() && static_cast<int>( type_ids->size() ) != n_atoms )
        {
            throw std::runtime_error(
                "Mismatch: type_ids has size " + std::to_string( type_ids->size() )
                + " but the number of atoms inferred is " + std::to_string( n_atoms ) );
        }

        const auto L   = box.get_lattice();
        const auto pbc = box.pbc;
        for( int i = 0; i < 3; i++ )
        {
            if( pbc[i] )
            {
                if( rc > 0.5 * L[i] )
                {
                    throw std::runtime_error(
                        "Rc is too large. In periodic directions, it cannot be larger than half the box size." );
                }
            }
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

    std::optional<std::vector<int>> type_ids = std::nullopt;
    Eigen::MatrixXd epsilon_matrix{};
    Eigen::MatrixXd sigma_matrix{};

    SimulationBoxInfo box{};
    NeighbourListIndices neighbour_indices{};
    NeighbourListImages neighbour_images{};

    LennardJones( double sigma, double epsilon, double rc, double ro )
            : sigma( sigma ), epsilon( epsilon ), rc( rc ), ro( ro )
    {
    }

    LennardJones(
        double sigma, double epsilon, double rc, double ro, const std::vector<int> & type_ids,
        const ParameterLookupMap & parameter_map )
            : LennardJones( sigma, epsilon, rc, ro )
    {
        std::vector<int> type_labels = type_ids;
        std::sort( type_labels.begin(), type_labels.end() );
        type_labels.erase( std::unique( type_labels.begin(), type_labels.end() ), type_labels.end() );

        const auto n_types = static_cast<Eigen::Index>( type_labels.size() );
        epsilon_matrix     = Eigen::MatrixXd::Constant( n_types, n_types, epsilon );
        sigma_matrix       = Eigen::MatrixXd::Constant( n_types, n_types, sigma );

        for( Eigen::Index i = 0; i < n_types; ++i )
        {
            for( Eigen::Index j = 0; j < n_types; ++j )
            {
                const auto parameter = parameter_map.find( { type_labels[i], type_labels[j] } );
                if( parameter != parameter_map.end() )
                {
                    epsilon_matrix( i, j ) = parameter->second.first;
                    sigma_matrix( i, j )   = parameter->second.second;
                }
            }
        }

        this->type_ids.emplace();
        this->type_ids->reserve( type_ids.size() );
        for( const int type_id : type_ids )
        {
            const auto compact_id = std::lower_bound( type_labels.begin(), type_labels.end(), type_id );
            this->type_ids->push_back( static_cast<int>( compact_id - type_labels.begin() ) );
        }
    }

    void recompute_neighbour_lists( const Eigen::Ref<Vectorfield> positions );

    Vectorfield compute_virial( const Eigen::Ref<Vectorfield> positions );

    double energy_and_forces( const Eigen::Ref<Vectorfield> positions, Eigen::Ref<Vectorfield> forces );
};
} // namespace Calci
