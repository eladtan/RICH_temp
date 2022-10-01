#include "../../../../pybind/include.h"
#include "../Tetrahedron.hpp"

#define MODULE_NAME tetrahedron
#define CLASS_NAME "Tetrahedron"
#define MODULE_VERSION "1.0"

static void __exportTetrahedron(py::module &module)
{
    module.doc() = "A class describing tetrahedra";
    py::class_<Tetrahedron>(module, CLASS_NAME)
        .def(py::init<>(), "Null constructor")
        .def(py::init<Tetrahedron const&>(), "Copy constructor")
        .def(PYTHON_ASSIGN_OPERATOR, &Tetrahedron::operator=, py::is_operator(), "Copy assignment");
}

PYBIND11_MODULE(MODULE_NAME, module)
{
    module.doc() = std::string("This module contains the class '") + CLASS_NAME + std::string("'.");
    __exportTetrahedron(module);
    module.attr(PYTHON_VERSION_ATTR) = (PYBIND_DEVELOPING == 1)? PYBIND_DEVELOPING_VERSION : std::string(MODULE_VERSION);
}