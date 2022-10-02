#include "../../../../pybind/include.h"
#include "../Intersections.hpp"

#define MODULE_NAME intersections
#define CLASS_NAME "Sphere"
#define MODULE_VERSION "1.0"

static void __exportSphere(py::module &module)
{
    py::class_<Sphere>(module, CLASS_NAME, "Sphere data")
        .def(py::init<>(), "Default constructor");
}

PYBIND11_MODULE(MODULE_NAME, module)
{
    module.doc() = std::string("This module contains the class '") + CLASS_NAME + std::string("'.");
    __exportSphere(module);
    module.def("face_sphere_intersections", &FaceSphereIntersections, py::arg("face"), py::arg("sphere"), py::arg("normal"));
    module.attr(PYTHON_VERSION_ATTR) = (PYBIND_DEVELOPING == 1)? PYBIND_DEVELOPING_VERSION : std::string(MODULE_VERSION);
}