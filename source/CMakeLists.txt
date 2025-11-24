cmake_minimum_required(VERSION 3.20.2)
project(rich)
enable_language(Fortran CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (${MPI})
    message("Compiling with MPI")
else()
    message("Compiling without MPI")
endif()

message(STATUS "CXX Compiler: " ${CMAKE_CXX_COMPILER})
message(STATUS "C Compiler: " ${CMAKE_C_COMPILER})
message(STATUS "Fortran Compiler: " ${CMAKE_Fortran_COMPILER})
message(STATUS "CXX Standard: " ${CMAKE_CXX_STANDARD})
message(STATUS "Build Type: " ${CMAKE_BUILD_TYPE})

message(STATUS "CMAKE_CXX_FLAGS = " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_CXX_FLAGS_RELEASE = " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "CMAKE_CXX_FLAGS_DEBUG = " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "CMAKE_Fortran_FLAGS = " ${CMAKE_Fortran_FLAGS})
message(STATUS "CMAKE_Fortran_FLAGS_RELEASE = " ${CMAKE_Fortran_FLAGS_RELEASE})
message(STATUS "CMAKE_Fortran_FLAGS_DEBUG = " ${CMAKE_Fortran_FLAGS_DEBUG})
message(STATUS "Root_dir = " ${PROJECT_ROOT_DIR})

# ADDING PROFILER
if (${PROF})
    message(STATUS "Compiling with GProf")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pg")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -pg")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -pg")
else()
    message(STATUS "Compiling without GProf")
endif()

message(STATUS "Adding linker -flto option")
add_link_options(-flto)

set(proj_dir "${PROJECT_SOURCE_DIR}")

find_package(pybind11 QUIET REQUIRED PATHS ${PYBIND11} NO_DEFAULT_PATH)
if(pybind11_FOUND)
    message(STATUS "pybind11 directory: " ${pybind11_DIR})
    message(STATUS "python exe path: " ${PYTHON_EXECUTABLE})
else()
    message(STATUS "PYBIND11 NOT FOUND")
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
include_directories("${BOOST_DIR}/include")
include_directories(${HDF5_INCLUDE})
include_directories(${R3D_INCLUDE})
include_directories("${PYBIND11_DIR}/include")

message("BOOST DIRECTORY ${BOOST_DIR}")
message("proj_dir ${proj_dir}")
message("HDF5 ${HDF5_LIBS}")
message("VTK ${VTK_DIR}")

# vcl
if(VCL_INCLUDE)
    message(STATUS "VCL ${VCL_INCLUDE}")
    include_directories("${VCL_INCLUDE}")
    add_compile_definitions("USE_VCL_VECTORIZATION")
endif()

# clipper
if(CLIPPER_INCLUDE)
    message(STATUS "CLIPPER ${CLIPPER_INCLUDE}")
    include_directories("${CLIPPER_INCLUDE}")
    add_compile_definitions("USE_CLIPPER")
endif()

# python
if(PYTHON_INCLUDE)
    if(PYTHON_LIB_DIRECTORY)
        # target json_cpp
        message(STATUS "JSON_CPP ${PYTHON_INCLUDE}, LIB ${PYTHON_LIB_DIRECTORY}")
        include_directories(${PYTHON_INCLUDE})
        
        find_library(PYTHON_LIB NAMES python3.9 libpython3.9 PATHS ${PYTHON_LIB_DIRECTORY} NO_DEFAULT_PATH)
        message(STATUS "PYTHON_LIB ${PYTHON_LIB} (DIR ${PYTHON_LIB_DIRECTORY})")
    else()
        message(STATUS "NOT INCLUDING PYTHON, LIB IS EMPTY (EVEN THOUGH INCLUDE IS NOT)")
    endif()
else()
    message(STATUS "NO PYTHON")
endif()

# json
if(JSONCPP_INCLUDE)
    if(JSONCPP_LIB_DIRECTORY)
        # target json_cpp
        message(STATUS "JSON_CPP ${JSONCPP_INCLUDE}, LIB ${JSONCPP_LIB_DIRECTORY}")
        include_directories(${JSONCPP_INCLUDE})
        
        find_library(JSONCPP_LIB NAMES jsoncpp libjsoncpp PATHS ${JSONCPP_LIB_DIRECTORY} NO_DEFAULT_PATH)
        message(STATUS "JSON_CPP_LIB ${JSONCPP_LIB} (DIR ${JSONCPP_LIB_DIRECTORY})")
    else()
        message(STATUS "NOT INCLUDING JSON_CPP, LIB IS EMPTY (EVEN THOUGH INCLUDE IS NOT)")
    endif()
else()
    message(STATUS "NO JSON")
endif()

# cgal
if(DEFINED CGAL_INCLUDE)
    message(STATUS "CGAL_INCLUDE ${CGAL_INCLUDE}")
    include_directories(${CGAL_INCLUDE})
else()
    message(STATUS "NO JSON")
endif()

# vtune
if(DEFINED VTUNE_INCLUDE)
    # target json_cpp
    message(STATUS "VTUNE ${VTUNE_INCLUDE}")
    include_directories(${VTUNE_INCLUDE})
else()
    message(STATUS "NO VTUNE")
endif()

# Build Core Static Library
message(STATUS "CMaking rich_core static library")
add_library(rich_core STATIC ${SRC_R3D} ${SRC_CPP})
target_link_libraries(
    rich_core 
    PUBLIC
    ${HDF5_LIBS} 
    ${VTK_LIBRARIES}
    ${JSONCPP_LIB}
    "-lstdc++"
)

add_executable(${EXE_NAME} ${TEST_DIR})

target_link_libraries(
    ${EXE_NAME} 
    PRIVATE
    rich_core
)

# link boost
if(DEFINED BOOST_DIR)
    target_link_directories(${EXE_NAME} PRIVATE ${BOOST_DIR}/lib)
endif()

# link pybind11
if(pybind11_FOUND)
    target_link_libraries(${EXE_NAME} PRIVATE pybind11::embed pybind11::module)
endif()

# link json cpp
if(DEFINED JSONCPP_LIB_DIRECTORY)
    target_link_libraries(${EXE_NAME} PRIVATE ${JSONCPP_LIB})
endif()

# link json cpp
if(DEFINED PYTHON_LIB_LIBRARY)
    target_link_libraries(${EXE_NAME} PRIVATE ${PYTHON_LIB})
endif()

# link vtune
if((DEFINED VTUNE_INCLUDE) AND (DEFINED VTUNE_LIB_DIRECTORY))
    set(VTUNE_LIB ${VTUNE_LIB_DIRECTORY}/libittnotify.a)
    message("VTUNE_LIB ${VTUNE_LIB}")
    target_link_libraries(${EXE_NAME} PRIVATE "${VTUNE_LIB}")
    add_compile_definitions("WITH_VTUNE")
endif()

vtk_module_autoinit(
  TARGETS ${EXE_NAME} rich_core
  MODULES ${VTK_LIBRARIES}
)

add_subdirectory(../tests_catch2 tests_catch2)
