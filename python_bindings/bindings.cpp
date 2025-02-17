#include <calci/defines.hpp>
#include <calci/utils.hpp>
#include <calci/backend.hpp>


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
    m.def( "check_constraints", &Calci::mic, "Wrapped vector obtained from minimum image convention." );
    m.def( "set_num_threads", &Calci::Backend::set_num_threads );
    m.def( "get_num_threads", &Calci::Backend::get_num_threads );
}