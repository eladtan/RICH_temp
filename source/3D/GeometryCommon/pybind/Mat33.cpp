#include "../../../../pybind/include.h"
#include "../Mat33.hpp"

#define MODULE_NAME mat33
#define CLASS_NAME std::string("Mat33")
#define MODULE_VERSION "1.0"

template<typename T>
void __templatedExportMat33(py::module &module, std::string &&name)
{
    using Matrix = Mat33<T>;
    std::string className = CLASS_NAME + std::string("_") + name;
    py::class_<Matrix>(module, className.c_str())
        .def(py::init<>())
        .def(py::init<T, T, T, T, T, T, T, T, T>())
        .def(py::init<const Matrix&>())
        .def("at", py::overload_cast<int, int>(&Matrix::at), "Returns the element at (row, col)")
        .def("at", py::overload_cast<int, int>(&Matrix::at, py::const_), "Returns the element at (row, col) - const version")
        .def(PYTHON_CALL_OPERATOR, py::overload_cast<int, int>(&Matrix::operator()), "Returns the element at (row, col)")
        .def(PYTHON_CALL_OPERATOR, py::overload_cast<int, int>(&Matrix::operator(), py::const_), "Returns the element at (row, col) - const version")
        .def("determinant", &Matrix::determinant, "Returns the determinant of the matrix")
        .def("det", &Matrix::determinant, "Returns the determinant of the matrix")
        .def("inverse", &Matrix::inverse, "Returns the inverse of the matrix")
        .def("inv", &Matrix::inverse, "Returns the inverse of the matrix")
        .def("transpose", &Matrix::transpose, "Returns the transpose of the matrix")
        .def("trans", &Matrix::transpose, "Returns the transpose of the matrix")
        .def(PYTHON_ASSIGN_OPERATOR, py::overload_cast<const Matrix&>(&Matrix::operator=), "Assignment");
}

PYBIND11_MODULE(MODULE_NAME, module)
{
    module.doc() = std::string("This module contains the classes '") + CLASS_NAME + std::string("_int' and '") + CLASS_NAME + std::string("_float'.");
   __templatedExportMat33<int>(module, "int");
   __templatedExportMat33<float>(module, "float");
    module.attr(PYTHON_VERSION_ATTR) = (PYBIND_DEVELOPING == 1)? "dev" : std::string(MODULE_VERSION);
}