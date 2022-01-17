#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "meshing.h"

PYBIND11_MODULE(_pcdmeshing, m) {
    m.doc() = "Point Cloud meshing via CGAL";

    m.def(
        "meshing_from_paths", &meshing_from_paths,
        py::arg("input_path"),
        py::arg("output_path"),
        py::arg("max_edge_length") = 0.,
        py::call_guard<py::gil_scoped_release>(),
        "Create a mesh from a pointcloud on disk"
    );

    m.def(
        "meshing_from_paths_with_vis", &meshing_with_visibility_from_paths,
        py::arg("input_path"),
        py::arg("output_path"),
        py::arg("endpoints_path"),
        py::arg("observations_path"),
        py::arg("max_edge_length") = 0.,
        py::arg("max_visibility") = 0,
        py::arg("post_filtering") = true,
        py::call_guard<py::gil_scoped_release>(),
        "Create a mesh using point visibility from a pointcloud on disk"
    );
}
