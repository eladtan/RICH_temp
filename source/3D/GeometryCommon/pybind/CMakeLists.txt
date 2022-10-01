cmake_minimum_required(VERSION 3.0)
project(tests LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 17)     # required version 14 or above, in order to compile
find_package(pybind11 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")  # todo remove that and replace it with a condition

set(PYBIND_LIBRARIES pybind11::embed pybind11::module)
set(TETRAHEDRON_PYBIND_SRC ../Tetrahedron.cpp)
set(VECTOR3D_PYBIND_SRC ../../../misc/utils.cpp ../../../misc/serializable.cpp ../Vector3D.cpp)
set(FACE_PYBIND_SRC ${VECTOR3D_PYBIND_SRC} ../Face.cpp)
set(MAT33_PYBIND_SRC "")
set(MAT44_PYBIND_SRC "")

pybind11_add_module(tetrahedron ${TETRAHEDRON_PYBIND_SRC} Tetrahedron.cpp)
target_link_libraries(tetrahedron PRIVATE ${PYBIND_LIBRARIES})

pybind11_add_module(face ${FACE_PYBIND_SRC} Face.cpp)
target_link_libraries(face PRIVATE ${PYBIND_LIBRARIES} ${Boost_INCLUDE_DIRS})

pybind11_add_module(mat33 ${MAT33_PYBIND_SRC} Mat33.cpp)
target_link_libraries(mat33 PRIVATE ${PYBIND_LIBRARIES} ${Boost_INCLUDE_DIRS})

pybind11_add_module(mat44 ${MAT44_PYBIND_SRC} Mat44.cpp)
target_link_libraries(mat44 PRIVATE ${PYBIND_LIBRARIES})

pybind11_add_module(vector3D ${VECTOR3D_PYBIND_SRC} Vector3D.cpp)
target_link_libraries(vector3D PRIVATE ${PYBIND_LIBRARIES} ${Boost_INCLUDE_DIRS})