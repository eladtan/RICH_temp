#include "../../../../pybind/include.h"
#include "../Vector3D.hpp"

#define MODULE_NAME vector3D
#define CLASS_NAME "Vector3D"
#define MODULE_VERSION "1.0"

void __ExportVector3D(py::module &module)
{
    // todo - serializable
    py::class_<Vector3D/*, Serializable*/>(module, CLASS_NAME, "A class representing a 3D Mathematical vector")
        .def(py::init<>(), "Null constructor")
        .def(py::init<double, double, double>(), "Default constructor")
        .def(py::init<const Vector3D&>(), "Copy constructor")
        .def("set", &Vector3D::Set, "Set vector components")
        .def(PYTHON_BRACKET_GET_OPERATOR, py::overload_cast<size_t>(&Vector3D::operator[]), py::is_operator(), "Indexed access to member")
        .def(PYTHON_BRACKET_GET_OPERATOR, py::overload_cast<size_t>(&Vector3D::operator[], py::const_), py::is_operator(), "Indexed access to member - const version")
        .def(PYTHON_BRACKET_SET_OPERATOR, [](Vector3D *vec, size_t index, double val){(*vec)[index] = val;}, py::is_operator(), "Indexed Assignment")
        .def(PYTHON_PEQ_OPERATOR, &Vector3D::operator+=, py::is_operator(), "Addition to current vector")
        .def(PYTHON_MEQ_OPERATOR, &Vector3D::operator-=, py::is_operator(), "Subtraction to current vector")
        .def(PYTHON_ASSIGN_OPERATOR, &Vector3D::operator=, py::is_operator(), "Assignment operator")
        .def(PYTHON_MULEQ_OPERATOR, &Vector3D::operator*=, py::is_operator(), "Multiplication by a given scalar")
        .def(PYTHON_EQ_OPERATOR, &Vector3D::operator==, py::is_operator(), "Compare 3D-Vectors (up to an arbitrary precision)")
        .def("rotate_X", &Vector3D::RotateX, "Rotates the vector around the X axis")
        .def("rotate_Y", &Vector3D::RotateY, "Rotates the vector around the Y axis")
        .def("rotate_Z", &Vector3D::RotateZ, "Rotates the vector around the Z axis")
        .def("round", &Vector3D::Round, "Integer round of the vector's entries");
}

void __ExportAdditionalFunctions(py::module &module)
{
    module.def("abs", py::overload_cast<const Vector3D&>(&abs), "Returns the norm of a vector");
    module.def("fastabs", &fastabs, "Returns the norm of a vector, less accurate");
    module.def("fast_abs", &fastabs, "Returns the norm of a vector, less accurate"); // added
    module.def(PYTHON_ADD_OPERATOR, &operator+, py::is_operator(),
                "Given two vectors, returns a new vector which is term by term addition");
    module.def(PYTHON_SUB_OPERATOR, &operator-, py::is_operator(),
                "Given two vectors, returns a new vector which is term by term subtraction");
    module.def(PYTHON_MUL_OPERATOR, py::overload_cast<double, const Vector3D&>(&operator*), py::is_operator(),
                "Given a vector and a scalar, returns a new vector which is the multiplication of the vector by the scalar (multiplication from left)");
    module.def(PYTHON_MUL_OPERATOR, py::overload_cast<const Vector3D&, double>(&operator*), py::is_operator(),
                "Given a vector and a scalar, returns a new vector which is the multiplication of the vector by the scalar (multiplication from right)");
    module.def(PYTHON_DIV_OPERATOR, &operator/,
                "Given a vector and a scalar, returns a new vector which is the given vector divided by the scalar");
    module.def("scalar_prod", &ScalarProd, "Returns the scalar product of two given vectors");
    module.def("scalar_product", &ScalarProd, "Returns the scalar product of two given vectors"); // added
    module.def("scap", &ScalarProd, "Returns the scalar product of two given vectors");  // added
    module.def("calc_angle", &CalcAngle, "Returns the angle between two given vectors (in radians)");
    module.def("angle", &CalcAngle, "Returns the angle between two given vectors (in radians)"); // added
    module.def("projection", &Projection, "Calculates the projection of one vector in the direction of the second");
    module.def("rotate_X", &RotateX,
                "Returns a new vector which is the rotation of a given vector, in a given angle (in radians), around the X axis");
    module.def("rotate_Y", &RotateY,
                "Returns a new vector which is the rotation of a given vector, in a given angle (in radians), around the Y axis");
    module.def("rotate_Z", &RotateZ,
                "Returns a new vector which is the rotation of a given vector, in a given angle (in radians), around the Z axis");
    module.def("reflect", &Reflect, "Returns a new vector which is a reflection of a given vector, by a plane which is given by its normal");
    module.def("distance", &distance, "Returns the distance between the two given vectors");
    module.def("dist", &distance, "Returns the distance between the two given vectors"); // added
    module.def("cross_product", py::overload_cast<const Vector3D&, const Vector3D&>(&CrossProduct),
                "Returns a new vector which is a cross product of the two given vectors");
    module.def("cross_prod", py::overload_cast<const Vector3D&, const Vector3D&>(&CrossProduct),
                "Returns a new vector which is a cross product of the two given vectors"); // added
    module.def("cross_product", py::overload_cast<const Vector3D&, const Vector3D&, Vector3D&>(&CrossProduct),
                "Sets the cross product of the two given vectors into a third given vector");
    module.def("split", &Split, "Splits a vector of 3D points to components");
    module.def("normalize", &normalize, "Returns a new vector which is the normalization of the given vector");
}

PYBIND11_MODULE(MODULE_NAME, module)
{
    module.doc() = std::string("This module contains the class '") + CLASS_NAME + std::string("'.");
   __ExportVector3D(module);
   __ExportAdditionalFunctions(module);
    module.attr(PYTHON_VERSION_ATTR) = (PYBIND_DEVELOPING == 1)? PYBIND_DEVELOPING_VERSION : std::string(MODULE_VERSION);
}