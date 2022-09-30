cmake_minimum_required(VERSION 3.0)
project(test_Mat33 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 14)
find_package(pybind11 REQUIRED)

pybind11_add_module(mat33 Mat33.cpp)
pybind11_add_module(mat44 Mat44.cpp)
pybind11_add_module(tetrahedron Tetrahedron.cpp ../Tetrahedron.cpp)
pybind11_add_module(face Face.cpp ../Face.cpp)
