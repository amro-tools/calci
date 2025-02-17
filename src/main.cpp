#ifdef _OPENMP
#include <omp.h>
#endif

#include <chrono>
#include <calci/defines.hpp>
#include <calci/utils.hpp>
#include <format>
#include <iostream>

int get_n_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1.0;
#endif
}

int main()
{
    int n_molecules = { 5000 };
    int n_atoms     = { 3 * n_molecules };

    double r_OH = 1;
    // double r_HH = 1.5;

    double mO = 16.0;
    double mH = 1.0;

    const int maxiter      = 500;
    const double tolerance = 1e-10;
    const std::array<bool, 3> pbc{ false, false, false };
    const Calci::Vector3 cell_lengths = { 20.0, 20.0, 20.0 };

    Calci::Vectorfield positions( n_atoms, 3 );
    Calci::Vectorfield velocities( n_atoms, 3 );
    Calci::Scalarfield masses( n_atoms );

    // Define positions
    for( int i = 0; i < n_molecules; i++ )
    {
        const int idx_O  = 3 * i;
        const int idx_H1 = idx_O + 1;
        const int idx_H2 = idx_O + 2;

        masses( idx_O )  = mO;
        masses( idx_H1 ) = mH;
        masses( idx_H2 ) = mH;

        positions.row( idx_O )  = 2.0 * Calci::Vector3::Random();
        positions.row( idx_H1 ) = positions.row( idx_O ) + r_OH * 1.4 * Calci::Vector3::Random().normalized();
        positions.row( idx_H2 ) = positions.row( idx_O ) + r_OH * 1.15 * Calci::Vector3::Random().normalized();
    }

    // Define velocities
    for( int i = 0; i < n_molecules; i++ )
    {
        const int idx_O  = 3 * i;
        const int idx_H1 = idx_O + 1;
        const int idx_H2 = idx_O + 2;

        velocities.row( idx_O )  = 0.1 * Calci::Vector3::Random();
        velocities.row( idx_H1 ) = 0.1 * Calci::Vector3::Random();
        velocities.row( idx_H2 ) = 0.1 * Calci::Vector3::Random();
    }

    std::cout << std::format( "{} threads\n", get_n_threads() );

    Calci::Vector3 dr = positions.row(0) - positions.row(1);
    Calci::Vector3 wrapped_dr = Calci::mic( dr, cell_lengths, pbc );

    std::cout << std::format( " wrapped distance = {}\n", dr.norm() );
}