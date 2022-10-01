#pragma once
#ifndef PYBIND11_INCLUDE
#define PYBIND11_INCLUDE 1

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <string>
#include <iostream>

#define PYTHON_CALL_OPERATOR "__call__"             // ()
#define PYTHON_ADD_OPERATOR "__add__"               // +
#define PYTHON_SUB_OPERATOR "__sub__"               // -
#define PYTHON_MUL_OPERATOR "__mul__"               // *
#define PYTHON_DIV_OPERATOR "__div__"               // /
#define PYTHON_MOD_OPERATOR "__mod__"               // %
#define PYTHON_NEG_OPERATOR "__neg__"               // ~
#define PYTHON_EQ_OPERATOR "__eq__"                 // ==
#define PYTHON_BRACKET_GET_OPERATOR "__getitem__"   // []
#define PYTHON_BRACKET_SET_OPERATOR "__setitem__"   // [] = ...
#define PYTHON_ASSIGN_OPERATOR "assign"             // =
#define PYTHON_PEQ_OPERATOR "__iadd__"              // +=
#define PYTHON_MEQ_OPERATOR "__isub__"              // -=
#define PYTHON_MULEQ_OPERATOR "__imul__"            // *=

/**
The name of an attribute discribing the version of each module.
Can be accessed by module.__version__ if 'module' is the imported module.
will be showed "dev" (PYBIND_DEVELOPING_VERSION) if current developing (PYBIND_DEVELOPING)
*/
#define PYTHON_VERSION_ATTR "__version__"
#define PYBIND_DEVELOPING_VERSION "dev"

#define PYBIND_DEVELOPING 1
#define PYBIND_DEBUG 0

namespace py = pybind11;
using namespace pybind11::literals;

/*
void pybind_message(std::string &&str)
{
    std::cout << "PYBIND11_GENERATOR: \t" << str << std::endl;
}
*/

#endif