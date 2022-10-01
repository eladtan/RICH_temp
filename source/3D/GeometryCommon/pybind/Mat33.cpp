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
    std::string classDoc = std::string("A very simple class for a 3x3 matrix of ") + name;
    py::class_<Matrix>(module, className.c_str(), classDoc.c_str())
        .def(py::init<>(), "Null constructor")
        .def(py::init<T, T, T, T, T, T, T, T, T>(), "Default constructor")
        .def(py::init<const Matrix&>(), "Copy constructor")
        .def("at", py::overload_cast<int, int>(&Matrix::at), "Returns the element at (row, col)")
        .def("at", py::overload_cast<int, int>(&Matrix::at, py::const_), "Returns the element at (row, col) - const version")
        .def(PYTHON_CALL_OPERATOR, py::overload_cast<int, int>(&Matrix::operator()), py::is_operator(), "Returns the element at (row, col)")
        .def(PYTHON_CALL_OPERATOR, py::overload_cast<int, int>(&Matrix::operator(), py::const_), py::is_operator(), "Returns the element at (row, col) - const version")
        .def("determinant", &Matrix::determinant, "Returns the determinant of the matrix")
        .def("det", &Matrix::determinant, "Returns the determinant of the matrix") // added
        .def("inverse", &Matrix::inverse, "Returns the inverse of the matrix")
        .def("inv", &Matrix::inverse, "Returns the inverse of the matrix") // added
        .def("transpose", &Matrix::transpose, "Returns the transpose of the matrix")
        .def("t", &Matrix::transpose, "Returns the transpose of the matrix") // added
        .def(PYTHON_ASSIGN_OPERATOR, py::overload_cast<const Matrix&>(&Matrix::operator=), py::is_operator(), "Assignment");
}

PYBIND11_MODULE(MODULE_NAME, module)
{
    module.doc() = std::string("This module contains the classes '") + CLASS_NAME + std::string("_int' and '") + CLASS_NAME + std::string("_float'.");
   __templatedExportMat33<int>(module, "int");
   __templatedExportMat33<float>(module, "float");
    module.attr(PYTHON_VERSION_ATTR) = (PYBIND_DEVELOPING == 1)? PYBIND_DEVELOPING_VERSION : std::string(MODULE_VERSION);
}