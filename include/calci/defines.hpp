#pragma once
#include <Eigen/Core>

namespace Calci
{

using Vector3 = Eigen::Matrix<double, 1, 3>;

// Row-major for compatibility with numpy arrays. See https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html
using Vectorfield = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::StorageOptions::RowMajor>;
using Scalarfield = Eigen::Matrix<double, Eigen::Dynamic, 1>;

} // namespace Calci