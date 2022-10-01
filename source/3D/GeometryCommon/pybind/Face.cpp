#include "Vector3D.h"
#include "../Face.hpp"

#define MODULE_NAME face
#define CLASS_NAME "Face"
#define MODULE_VERSION "1.0"

void __ExportFace(py::module &module)
{
    py::class_<Face>(module, CLASS_NAME, "A class describing a face between cells")
            .def(py::init<>(), "Default constructor")
            .def(py::init<const point_vec_v&, std::size_t, std::size_t>(), "Default constructor")
            .def(py::init<const Face&>(), "Copy constructor")
            .def("get_area", &Face::GetArea, "Returns the area of the face")
            .def(PYTHON_ASSIGN_OPERATOR, py::overload_cast<const Face&>(&Face::operator=), py::is_operator(), "Assignment");
}

PYBIND11_MODULE(MODULE_NAME, module)
{
    module.doc() = std::string("This module contains the class '") + CLASS_NAME + std::string("'.");
    __ExportFace(module);
    module.def("calc_centroid", &calc_centroid, "Calculates the centroid of a face");
    module.attr(PYTHON_VERSION_ATTR) = (PYBIND_DEVELOPING == 1)? PYBIND_DEVELOPING_VERSION : std::string(MODULE_VERSION);
}