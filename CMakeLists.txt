cmake_minimum_required(VERSION 3.20)
project(RICH LANGUAGES C CXX Fortran VERSION 1.0.0)

# Options
option(ENABLE_MPI "Enable MPI support" OFF)
option(ENABLE_VTK "Enable VTK output support" OFF)
option(ENABLE_PROFILING "Enable profiling" OFF)
set(ENERGY_GROUPS_NUM "1" CACHE STRING "Energy groups for multigroup radiation")
set(PROBLEM_NAME "sedov2d_test" CACHE STRING "Problem name (directory in runs/)")

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_compile_definitions(ENERGY_GROUPS_NUM=${ENERGY_GROUPS_NUM})

message(STATUS "==============================================")
message(STATUS "RICH Configuration")
message(STATUS "==============================================")
message(STATUS "Build Type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "Compilers:    ${CMAKE_C_COMPILER}, ${CMAKE_CXX_COMPILER}, ${CMAKE_Fortran_COMPILER}")
message(STATUS "Problem:      ${PROBLEM_NAME}")
message(STATUS "MPI:          ${ENABLE_MPI}")
message(STATUS "VTK:          ${ENABLE_VTK}")
message(STATUS "Profiling:    ${ENABLE_PROFILING}")
message(STATUS "Energy Groups: ${ENERGY_GROUPS_NUM}")
message(STATUS "==============================================")

# MPI configuration
if(ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS C CXX Fortran)
    add_compile_definitions(RICH_MPI)
    message(STATUS "MPI: ${MPI_C_COMPILER}, ${MPI_CXX_COMPILER}, ${MPI_Fortran_COMPILER}")
endif()

# Profiling
if(ENABLE_PROFILING)
    add_compile_options(-pg)
    add_link_options(-pg)
endif()

# Generate r3d config
set(R3D_MAX_VERTS 256)
set(SHIFT_POLY 1)
configure_file(
    ${CMAKE_SOURCE_DIR}/source/opt/r3d/config/r3d-config.h.in
    ${CMAKE_SOURCE_DIR}/source/opt/r3d/src/r3d-config.h
    @ONLY
)

# Find dependencies
find_package(OpenMP)
find_package(HDF5 REQUIRED COMPONENTS C CXX HL)

# VTK - optional, only when enabled
if(ENABLE_VTK)
    find_package(VTK REQUIRED COMPONENTS
        CommonCore CommonColor CommonDataModel CommonTransforms
        FiltersGeneral FiltersSources IOXML InteractionStyle)

    add_compile_definitions(RICH_VTK)
    message(STATUS "VTK support: Enabled")

    # VTK MPI components - optional, only when MPI is enabled
    if(ENABLE_MPI)
        find_package(VTK QUIET COMPONENTS ParallelMPI IOParallel IOParallelXML)
        if(VTK_ParallelMPI_FOUND)
            message(STATUS "VTK MPI support: Found")
            add_compile_definitions(RICH_VTK_MPI)
        else()
            message(STATUS "VTK MPI support: Not found (parallel VTU output will be disabled)")
        endif()
    endif()
else()
    message(STATUS "VTK support: Disabled (VTU output will not be available)")
endif()

find_package(Boost REQUIRED)

# pybind11 - check local first
if(EXISTS ${CMAKE_SOURCE_DIR}/opt/pybind11)
    list(PREPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/opt/pybind11)
endif()
find_package(pybind11 REQUIRED)

# Optional jsoncpp
find_package(jsoncpp QUIET)

# Source files
file(GLOB_RECURSE SRC_CPP "${CMAKE_SOURCE_DIR}/source/*.cpp")
file(GLOB_RECURSE PROBLEM_SRC "${CMAKE_SOURCE_DIR}/runs/${PROBLEM_NAME}/*.cpp")
file(GLOB SRC_R3D "${CMAKE_SOURCE_DIR}/source/opt/r3d/src/*.c")

if(NOT PROBLEM_SRC)
    message(FATAL_ERROR "No source files in runs/${PROBLEM_NAME}/")
endif()

# Executable
add_executable(rich ${PROBLEM_SRC} ${SRC_R3D} ${SRC_CPP})

set_target_properties(rich PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(rich PRIVATE
    ${CMAKE_SOURCE_DIR}/source
    ${CMAKE_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/source/opt/r3d/src
    ${CMAKE_SOURCE_DIR}/source/opt/vcl
    ${CMAKE_SOURCE_DIR}/source/opt/clipper
)

# Compile definitions
if(EXISTS ${CMAKE_SOURCE_DIR}/source/opt/clipper)
    target_compile_definitions(rich PRIVATE USE_CLIPPER)
endif()

# Link libraries
target_link_libraries(rich PRIVATE
    ${HDF5_LIBRARIES}
    pybind11::embed
)

if(ENABLE_VTK)
    target_link_libraries(rich PRIVATE ${VTK_LIBRARIES})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(rich PRIVATE OpenMP::OpenMP_CXX)
endif()

if(ENABLE_MPI)
    target_link_libraries(rich PRIVATE MPI::MPI_C MPI::MPI_CXX MPI::MPI_Fortran)
endif()

if(jsoncpp_FOUND)
    target_link_libraries(rich PRIVATE jsoncpp_lib)
endif()

# VTK initialization
if(ENABLE_VTK)
    vtk_module_autoinit(TARGETS rich MODULES ${VTK_LIBRARIES})
endif()

message(STATUS "Configuration complete - run 'make' to build")
