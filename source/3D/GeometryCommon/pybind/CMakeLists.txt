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

if(${BIND_TETRAHEDRON})
  set(MODULE "tetrahedron")
  message(STATUS "Binding Tetrahedron into module named ${MODULE}")
  pybind11_add_module(${MODULE} ${TETRAHEDRON_PYBIND_SRC} Tetrahedron.cpp)
  target_link_libraries(${MODULE} PRIVATE ${PYBIND_LIBRARIES})
endif()

if(${BIND_FACE})
  set(MODULE "face")
  message(STATUS "Binding Face into module named ${MODULE}")
  pybind11_add_module(${MODULE} ${FACE_PYBIND_SRC} Face.cpp)
  target_link_libraries(${MODULE} PRIVATE ${PYBIND_LIBRARIES} ${Boost_INCLUDE_DIRS})
endif()

if(${BIND_MAT33})
  set(MODULE "mat33")
  message(STATUS "Binding Mat33 into module named ${MODULE}")
  pybind11_add_module(${MODULE} ${MAT33_PYBIND_SRC} Mat33.cpp)
  target_link_libraries(${MODULE} PRIVATE ${PYBIND_LIBRARIES} ${Boost_INCLUDE_DIRS})
endif()

if(${BIND_MAT44})
  set(MODULE "mat44")
  message(STATUS "Binding Mat44 into module named ${MODULE}")
  pybind11_add_module(${MODULE} ${MAT44_PYBIND_SRC} Mat44.cpp)
  target_link_libraries(${MODULE} PRIVATE ${PYBIND_LIBRARIES})
endif()

if(${BIND_VECTOR3D})
  set(MODULE "vector3D")
  message(STATUS "Binding Vector3D into module named ${MODULE}")
  pybind11_add_module(${MODULE} ${VECTOR3D_PYBIND_SRC} Vector3D.cpp)
  target_link_libraries(${MODULE} PRIVATE ${PYBIND_LIBRARIES} ${Boost_INCLUDE_DIRS})
endif()