#include "calci/lennard_jones.hpp"
#include <calci/backend.hpp>
#include <calci/defines.hpp>
#include <calci/utils.hpp>

#define PYBIND11_DETAILED_ERROR_MESSAGES

// Bindings
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <format>

// Namespaces
using namespace std::string_literals; // For ""s
using namespace pybind11::literals;   // For ""_a
namespace py = pybind11;              // Convention

PYBIND11_MODULE( calcicpp, m )
{
    m.def( "set_num_threads", &Calci::Backend::set_num_threads );
    m.def( "get_num_threads", &Calci::Backend::get_num_threads );

    py::class_<Calci::SimulationBoxInfo>( m, "SimulationBoxInfo" )
        .def( py::init<>() )
        .def( py::init<const std::array<double, 3> &, const std::array<bool, 3> &>() )
        .def( "set_lattice", &Calci::SimulationBoxInfo::set_lattice )
        .def( "get_lattice", &Calci::SimulationBoxInfo::get_lattice )
        .def_readwrite( "pbc", &Calci::SimulationBoxInfo::pbc );

    py::class_<Calci::LennardJones>( m, "LennardJones" )
        .def( py::init<double, double, double, double>() )
        .def_readwrite( "sigma", &Calci::LennardJones::sigma )
        .def_readwrite( "epsilon", &Calci::LennardJones::epsilon )
        .def_readwrite( "rc", &Calci::LennardJones::rc )
        .def_readwrite( "ro", &Calci::LennardJones::ro )
        .def_readwrite( "verlet_skin_depth", &Calci::LennardJones::verlet_skin_depth )
        .def_readwrite( "box", &Calci::LennardJones::box )
        .def_readwrite( "neighbour_indices", &Calci::LennardJones::neighbour_indices )
        .def( "energy_and_forces", &Calci::LennardJones::energy_and_forces )
        .def( "recompute_neighbour_lists", &Calci::LennardJones::recompute_neighbour_lists );
}