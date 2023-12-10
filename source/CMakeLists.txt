cmake_minimum_required(VERSION 3.20.2)
project(rich)
enable_language(Fortran CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (${MPI})
    message("Compiling with MPI")
else()
    message("Compiling without MPI")
endif()

message("CXX Compiler: " ${CMAKE_CXX_COMPILER})
message("C Compiler: " ${CMAKE_C_COMPILER})
message("Fortran Compiler: " ${CMAKE_Fortran_COMPILER})
message("CXX Standard: " ${CMAKE_CXX_STANDARD})
message("Build Type: " ${CMAKE_BUILD_TYPE})

message("CMAKE_CXX_FLAGS = " ${CMAKE_CXX_FLAGS})
message("CMAKE_CXX_FLAGS_RELEASE = " ${CMAKE_CXX_FLAGS_RELEASE})
message("CMAKE_CXX_FLAGS_DEBUG = " ${CMAKE_CXX_FLAGS_DEBUG})
message("CMAKE_Fortran_FLAGS = " ${CMAKE_Fortran_FLAGS})
message("CMAKE_Fortran_FLAGS_RELEASE = " ${CMAKE_Fortran_FLAGS_RELEASE})
message("CMAKE_Fortran_FLAGS_DEBUG = " ${CMAKE_Fortran_FLAGS_DEBUG})
message("Root_dir = " ${PROJECT_ROOT_DIR})


# ADDING PROFILER
if (${PROF})
    message("Compiling with GProf")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pg")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -pg")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -pg")
else()
    message("Compiling without GProf")
endif()

set(proj_dir "${PROJECT_SOURCE_DIR}")

find_package(pybind11 QUIET REQUIRED PATHS ${PYBIND11} NO_DEFAULT_PATH)
if(pybind11_FOUND)
    message("pybind11 directory: " ${pybind11_DIR})
    message("python exe path: " ${PYTHON_EXECUTABLE})
else()
    message("PYBIND11 NOT FOUND")
endif()

# files to compile
file(GLOB_RECURSE SRC_CPP FOLLOW_SYMLINKS "${PROJECT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE TEST_DIR "${TEST_DIR}/*.cpp")
file(GLOB SRC_R3D "${PROJECT_SOURCE_DIR}/opt/r3d/src/*.c")

# HDF5
find_library(HDF5 hdf5 PATHS ${HDF5_LIB_DIRECTORY} NO_DEFAULT_PATH)
find_library(HDF5_HL hdf5_hl PATHS ${HDF5_LIB_DIRECTORY} NO_DEFAULT_PATH)
find_library(HDF5_CPP hdf5_cpp PATHS ${HDF5_LIB_DIRECTORY} NO_DEFAULT_PATH)
set(HDF5_LIBS ${HDF5_CPP} ${HDF5_HL} ${HDF5})

if (NOT HDF5_CPP)
    message(FATAL_ERROR "NO HDF5_CPP")
endif()
if (NOT HDF5_HL)
    message(FATAL_ERROR "NO HDF5_HL")
endif()
if (NOT HDF5)
    message(FATAL_ERROR "NO HDF5")
endif()

# VTK
# see examples here 
# https://vtk.org/doc/nightly/html/md__builds_gitlab-kitware-sciviz-ci_Documentation_Doxygen_ModuleMigration.html
find_package(VTK COMPONENTS 
    CommonCore 
    CommonColor
    CommonDataModel
    CommonTransforms
    FiltersGeneral
    FiltersSources
    IOXML 
    IOParallelXML 
    ParallelMPI 
    InteractionStyle
    NO_MODULE 
    PATHS ${VTK_DIRECTORY} NO_DEFAULT_PATH
)
if (NOT VTK_FOUND)
    message(FATAL_ERROR "NO VTK")
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_ROOT_DIR})
include_directories(${BOOST_INCLUDE})
include_directories(${HDF5_INCLUDE})
include_directories(${R3D_INCLUDE})
include_directories("${PYBIND11_DIR}/include")

message("proj_dir ${proj_dir}")
message("HDF5 ${HDF5_LIBS}")
message("VTK ${VTK_DIR}")

# vcl
if(DEFINED VCL_INCLUDE)
    message("VCL ${VCL_INCLUDE}")
    include_directories("${VCL_INCLUDE}")
    add_compile_definitions("USE_VCL_VECTORIZATION")
endif()

# json
if(DEFINED JSONCPP_INCLUDE)
    if(DEFINED JSONCPP_LIB_DIRECTORY)
        # target json_cpp
        message("JSON_CPP ${JSONCPP_INCLUDE}, LIB ${JSONCPP_LIB_DIRECTORY}")
        include_directories(${JSONCPP_INCLUDE})
        
        find_library(JSONCPP_LIB NAMES jsoncpp libjsoncpp PATHS ${JSONCPP_LIB_DIRECTORY} NO_DEFAULT_PATH)
        message("JSON_CPP_LIB ${JSONCPP_LIB} (DIR ${JSONCPP_LIB_DIRECTORY})")
    else()
        message("NOT INCLUDING JSON_CPP, LIB IS EMPTY (EVEN THOUGH INCLUDE IS NOT)")
    endif()
else()
    message("NO JSON")
endif()

# cgal
if(DEFINED CGAL_INCLUDE)
    message("CGAL_INCLUDE ${CGAL_INCLUDE}")
    include_directories(${CGAL_INCLUDE})
else()
    message("NO JSON")
endif()

add_executable(${EXE_NAME} ${TEST_DIR} ${SRC_R3D} ${SRC_CPP})

target_link_libraries(
    ${EXE_NAME} 
    PRIVATE
    ${HDF5_LIBS} 
    ${VTK_LIBRARIES}
    ${JSONCPP_LIB}
)

# link pybind11
if(pybind11_FOUND)
    target_link_libraries(${EXE_NAME} PRIVATE pybind11::embed pybind11::module)
endif()

# link json cpp
if(DEFINED JSONCPP_LIB_DIRECTORY)
    target_link_libraries(${EXE_NAME} PRIVATE ${JSONCPP_LIB})
endif()

vtk_module_autoinit(
  TARGETS ${EXE_NAME}
  MODULES ${VTK_LIBRARIES}
)
