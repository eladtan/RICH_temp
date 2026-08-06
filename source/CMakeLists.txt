cmake_minimum_required(VERSION 3.20.2)

if(NOT DEFINED CONFIG)
    message(FATAL_ERROR "Required config (-DCONFIG=): gnuReleaseMPI, gnuDebugMPI, intelReleaseMPI, intelDebugMPI, gnuRelease, gnuDebug, intelRelease, intelDebug")
endif()

include(${CMAKE_SOURCE_DIR}/../config/parse_arguments.cmake)
include(${CMAKE_SOURCE_DIR}/../config/set_compilers.cmake)
include(${CMAKE_SOURCE_DIR}/../config/compiler_flags.cmake)
include(${CMAKE_SOURCE_DIR}/../config/mixed.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_static_library.cmake)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(RICH_MONTECARLO_POLARIZATION "Enable Monte Carlo linear polarization packet state" OFF)
if(RICH_MONTECARLO_POLARIZATION)
    add_compile_definitions(MONTECARLO_POLARIZATION)
endif()

message(STATUS "CXX Standard: " ${CMAKE_CXX_STANDARD})
message(STATUS "Build Type: " ${CMAKE_BUILD_TYPE})

string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type_upper)
message(STATUS "CXX flags:     ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${build_type_upper}}")
message(STATUS "Fortran flags: ${CMAKE_Fortran_FLAGS} ${CMAKE_Fortran_FLAGS_${build_type_upper}}")

project(rich)
enable_testing()

set(EXE_NAME ${CMAKE_PROJECT_NAME}_${CONFIG})
enable_language(Fortran CXX)


# ADDING PROFILER
if (${PROF})
    message(STATUS "Compiling with GProf")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pg")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -pg")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -pg")
else()
    message(STATUS "Compiling without GProf")
endif()

if(NOT TEST_DIR)
    message(FATAL_ERROR "TEST_DIR is not set. Please set it (using '-DTEST_DIR=XX') to a test directory under 'runs/' or a repo-relative directory.")
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/../runs/${TEST_DIR}")
    set(TEST_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../runs/${TEST_DIR}")
elseif(EXISTS "${PROJECT_SOURCE_DIR}/../${TEST_DIR}")
    set(TEST_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../${TEST_DIR}")
else()
    message(FATAL_ERROR "Could not find test directory. Checked: ${PROJECT_SOURCE_DIR}/../runs/${TEST_DIR} and ${PROJECT_SOURCE_DIR}/../${TEST_DIR}")
endif()

# =============================================
# Submodule directory definitions
# =============================================
set(MADVORO_DIR      "${PROJECT_SOURCE_DIR}/3D/tessellation/voronoi")
set(MADCART_DIR      "${PROJECT_SOURCE_DIR}/3D/tessellation/cartesian")
set(STORM_DIR        "${PROJECT_SOURCE_DIR}/monte")
set(MPI_UTILS_DIR    "${PROJECT_SOURCE_DIR}/utils/mpi_utils")
set(SPATIAL_DS_DIR   "${PROJECT_SOURCE_DIR}/utils/spatial_ds")
set(MESH_DECOMP_DIR  "${PROJECT_SOURCE_DIR}/3D/tessellation/MeshDecomposer3D")
add_compile_definitions(RICH_IMC_DDMC_ENABLED)

# =============================================
# Source collection
# =============================================
# RICH sources are discovered recursively so adding a new RICH directory does
# not require editing this file. Dependency projects are excluded before the
# executable is created; their native CMake files below own their sources.
file(GLOB_RECURSE RICH_SRC
     "${PROJECT_SOURCE_DIR}/*.cpp"
     "${PROJECT_SOURCE_DIR}/*.c")

set(RICH_EXTERNAL_SOURCE_DIRS
    "${PROJECT_SOURCE_DIR}/monte"
    "${PROJECT_SOURCE_DIR}/3D/tessellation/voronoi"
    "${PROJECT_SOURCE_DIR}/3D/tessellation/cartesian"
    "${PROJECT_SOURCE_DIR}/3D/tessellation/MeshDecomposer3D"
    "${PROJECT_SOURCE_DIR}/utils/mpi_utils"
    "${PROJECT_SOURCE_DIR}/utils/spatial_ds"
    "${PROJECT_SOURCE_DIR}/utils/rma"
)
foreach(_external_source_dir IN LISTS RICH_EXTERNAL_SOURCE_DIRS)
    list(FILTER RICH_SRC EXCLUDE REGEX "^${_external_source_dir}(/|$)")
endforeach()

file(GLOB_RECURSE TEST_SRC CONFIGURE_DEPENDS
     "${TEST_SOURCE_DIR}/*.cpp"
     "${TEST_SOURCE_DIR}/*.c")
include(${CMAKE_SOURCE_DIR}/../config/find_HDF5.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_VTK.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_JSON.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_Boost.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_cgal.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_vtune.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_ParMETIS.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_ibverbs.cmake)
include(${CMAKE_SOURCE_DIR}/../config/include_3rdparty.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_pybind11.cmake)
include(${CMAKE_SOURCE_DIR}/../config/set_placeholders.cmake)

if(DEFINED PYBIND11_DIRECTORY)
    find_package(pybind11 QUIET CONFIG PATHS ${PYBIND11_DIRECTORY} NO_DEFAULT_PATH)
else()
    find_package(pybind11 QUIET CONFIG)
endif()

if(pybind11_FOUND)
    message(STATUS "pybind11 directory: ${pybind11_DIR}")
else()
    message(STATUS "pybind11 not found -- Python embedding disabled")
endif()

# HDF5
if(DEFINED DYNAMIC_LIBS)
    find_library(HDF5 hdf5 PATHS ${HDF5_LIB_DIRECTORY})
    find_library(HDF5_HL hdf5_hl PATHS ${HDF5_LIB_DIRECTORY})
    find_library(HDF5_CPP hdf5_cpp PATHS ${HDF5_LIB_DIRECTORY})
else()
    find_static_library(HDF5 hdf5 PATHS ${HDF5_LIB_DIRECTORY})
    find_static_library(HDF5_HL hdf5_hl PATHS ${HDF5_LIB_DIRECTORY})
    find_static_library(HDF5_CPP hdf5_cpp PATHS ${HDF5_LIB_DIRECTORY})
endif()
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


# Static HDF5 needs its transitive deps linked explicitly
if(NOT DEFINED DYNAMIC_LIBS)
    # Szip (compression filter used by this HDF5 build)
    if(DEFINED ENV{EBROOTSZIP})
        find_library(SZIP_LIB sz PATHS "$ENV{EBROOTSZIP}/lib" NO_DEFAULT_PATH)
    else()
        find_library(SZIP_LIB sz)
    endif()
    if(SZIP_LIB)
        list(APPEND HDF5_LIBS ${SZIP_LIB})
        message(STATUS "Szip for static HDF5: ${SZIP_LIB}")
    endif()
endif()

# If HDF5 was built with parallel support its headers include <mpi.h> and
# its library references MPI symbols.  Provide the MPI include path and
# link target even for serial builds so that compilation and linking succeed.
if(EXISTS "${HDF5_INCLUDE}/H5pubconf.h")
    file(STRINGS "${HDF5_INCLUDE}/H5pubconf.h" _h5_parallel
         REGEX "^#define H5_HAVE_PARALLEL ")
else()
    set(_h5_parallel "")
endif()

if(_h5_parallel)
    if(NOT DEFINED MPI)
        find_package(MPI QUIET)
        if(MPI_C_FOUND)
            include_directories(${MPI_C_INCLUDE_DIRS})
            if(NOT DEFINED DYNAMIC_LIBS)
                list(APPEND HDF5_LIBS MPI::MPI_C)
            endif()
            message(STATUS "HDF5 is parallel-enabled; adding MPI for serial build: ${MPI_C_INCLUDE_DIRS}")
        else()
            message(WARNING "HDF5 requires MPI headers (H5_HAVE_PARALLEL) but MPI was not found")
        endif()
    else()
        find_package(MPI REQUIRED)
        if(NOT DEFINED DYNAMIC_LIBS)
            list(APPEND HDF5_LIBS MPI::MPI_C)
        endif()
    endif()
endif()

# MPI (discovered in set_compilers.cmake via mpicxx wrapper)

# VTK (found and validated in config/find_VTK.cmake)

include_directories(${PROJECT_SOURCE_DIR})
include_directories("${PROJECT_SOURCE_DIR}/..") # TODO: REMOVE!
include_directories("${PROJECT_SOURCE_DIR}/utils")
include_directories("${PROJECT_SOURCE_DIR}/3D/tessellation")
include_directories("${BOOST_DIR}/include")
include_directories(${HDF5_INCLUDE})


# json
if(JSONCPP_INCLUDE AND JSONCPP_LIB_DIRECTORY)
    include_directories(${JSONCPP_INCLUDE})
    if(DEFINED DYNAMIC_LIBS)
        find_library(JSONCPP_LIB jsoncpp PATHS ${JSONCPP_LIB_DIRECTORY})
    else()
        # static library
        find_static_library(JSONCPP_LIB jsoncpp PATHS ${JSONCPP_LIB_DIRECTORY})
    endif()
endif()

# parmetis
if(PARMETIS_INCLUDE AND PARMETIS_LIB_DIRECTORY)
    include_directories(${PARMETIS_INCLUDE})
    if(DEFINED DYNAMIC_LIBS)
        find_library(PARMETIS_LIB parmetis PATHS ${PARMETIS_LIB_DIRECTORY})
    else()
        find_static_library(PARMETIS_LIB parmetis PATHS ${PARMETIS_LIB_DIRECTORY})
    endif()
endif()

# metis
if(METIS_INCLUDE AND METIS_LIB_DIRECTORY)
    include_directories(${METIS_INCLUDE})
    find_library(METIS_LIB metis PATHS ${METIS_LIB_DIRECTORY})
endif()

# gklib
if(GKLIB_INCLUDE AND GKLIB_LIB_DIRECTORY)
    include_directories(${GKLIB_INCLUDE})
    if(DEFINED DYNAMIC_LIBS)
        find_library(GKLIB_LIB GKlib PATHS ${GKLIB_LIB_DIRECTORY})
    else()
        find_static_library(GKLIB_LIB GKlib PATHS ${GKLIB_LIB_DIRECTORY})
    endif()
endif()

# cgal
if(CGAL_DIRECTORY)
    # TODO Later: crashes (MPFR)
    # find_package(CGAL QUIET PATHS ${CGAL_DIRECTORY} NO_DEFAULT_PATH)
endif()

# vtune
if(DEFINED VTUNE_INCLUDE)
    # target json_cpp
    include_directories(${VTUNE_INCLUDE})
endif()

# =============================================
# Parse MPI flags (reused for submodule targets)
# =============================================
if(DEFINED MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    separate_arguments(mpi_compile_flags UNIX_COMMAND "${MPI_COMPILE_FLAGS}")
    separate_arguments(mpi_link_flags UNIX_COMMAND "${MPI_LINK_FLAGS}")
endif()

# =============================================
# Dependency library targets
# =============================================
if(DEFINED MPI)
    set(MADVORO_WITH_MPI ON)
    set(MADCART_WITH_MPI ON)
    set(STORM_WITH_MPI ON)
else()
    set(MADVORO_WITH_MPI OFF)
    set(MADCART_WITH_MPI OFF)
    set(STORM_WITH_MPI OFF)
endif()
set(MADVORO_BUILD_EXAMPLES OFF)
set(MADCART_BUILD_EXAMPLES OFF)
set(STORM_BUILD_EXAMPLES OFF)
set(STORM_WITH_COMPTON ON)

set(MADVORO_DEPS_DIR "${PROJECT_SOURCE_DIR}/utils")
set(MADCART_DEPS_DIR "${PROJECT_SOURCE_DIR}/utils")
set(STORM_DEPS_DIR "${PROJECT_SOURCE_DIR}/utils")
set(STORM_MADVORO_DIR "${MADVORO_DIR}")
set(STORM_MADCART_DIR "${MADCART_DIR}")
set(STORM_MESH_DECOMPOSER_DIR "${MESH_DECOMP_DIR}")

add_subdirectory("${MPI_UTILS_DIR}" "${CMAKE_BINARY_DIR}/dependencies/mpi_utils"
                 EXCLUDE_FROM_ALL)
add_subdirectory("${MADVORO_DIR}" "${CMAKE_BINARY_DIR}/dependencies/madvoro"
                 EXCLUDE_FROM_ALL)
add_subdirectory("${MADCART_DIR}" "${CMAKE_BINARY_DIR}/dependencies/madcart"
                 EXCLUDE_FROM_ALL)
add_subdirectory("${STORM_DIR}" "${CMAKE_BINARY_DIR}/dependencies/storm"
                 EXCLUDE_FROM_ALL)

# Header-only dependencies do not currently define targets in their native
# CMake files, so expose their include trees as local interface targets.
add_library(spatial_ds INTERFACE)
target_include_directories(spatial_ds INTERFACE "${SPATIAL_DS_DIR}")

add_library(mesh_decomposer INTERFACE)
target_include_directories(mesh_decomposer INTERFACE "${MESH_DECOMP_DIR}")

message(STATUS "Dependency targets: madvoro madcart storm mpi_utils spatial_ds mesh_decomposer")

# =============================================
# RICH executable
# =============================================
add_executable(${EXE_NAME} ${TEST_SRC} ${RICH_SRC})

if(${MIXED_BUILD})
    message(STATUS "Mixed build: marking some files for debug")
    mark_debug_sources("${EXE_NAME}")
endif()

# ======================= LINKING =======================

if(CGAL_DIRECTORY)
    # TODO: at least this line
    # target_link_libraries(${EXE_NAME} PRIVATE CGAL::CGAL)

    # # Optionally: if you use CGAL Core
    # if(TARGET CGAL::CGAL_Core)
    #     target_link_libraries(my_program PRIVATE CGAL::CGAL_Core)
    # endif()
endif()

find_package(ZLIB REQUIRED)
target_link_libraries(
    ${EXE_NAME} 
    PRIVATE
    madvoro madcart storm mpi_utils spatial_ds mesh_decomposer
    ZLIB::ZLIB
    ${HDF5_LIBS} 
    ${VTK_LIBRARIES}
    ${CMAKE_DL_LIBS}
)
target_link_options(${EXE_NAME} PRIVATE -rdynamic)

if(DEFINED MPI)
    target_compile_options(${EXE_NAME} PRIVATE ${mpi_compile_flags})
    target_link_options(${EXE_NAME} PRIVATE ${mpi_link_flags})
endif()

if(DEFINED JSONCPP_LIB)
    target_link_libraries(${EXE_NAME} PRIVATE ${JSONCPP_LIB})
endif()

if(DEFINED PARMETIS_LIB)
    target_link_libraries(${EXE_NAME} PRIVATE ${PARMETIS_LIB})
    # define "WITH_PARMETIS"
    add_compile_definitions("WITH_PARMETIS")
endif()

if(DEFINED METIS_LIB)
    target_link_libraries(${EXE_NAME} PRIVATE ${METIS_LIB})
endif()

if(DEFINED GKLIB_LIB)
    target_link_libraries(${EXE_NAME} PRIVATE ${GKLIB_LIB})
endif()

# link boost
if(DEFINED BOOST_DIR)
    target_link_directories(${EXE_NAME} PRIVATE ${BOOST_DIR}/lib)
endif()

# link pybind11 (includes Python automatically)
if(pybind11_FOUND)
    target_link_libraries(${EXE_NAME} PRIVATE pybind11::embed)
endif()

# link vtune
if((DEFINED VTUNE_INCLUDE) AND (DEFINED VTUNE_LIB_DIRECTORY))
    set(VTUNE_LIB ${VTUNE_LIB_DIRECTORY}/libittnotify.a)
    target_link_libraries(${EXE_NAME} PRIVATE "${VTUNE_LIB}")
    add_compile_definitions("WITH_VTUNE")
endif()

if(DEFINED IBV)
    target_link_libraries(${EXE_NAME} PRIVATE ${IBVERBS_LIB})
    target_include_directories(${EXE_NAME} PRIVATE ${IBVERBS_INCLUDE})
endif()

vtk_module_autoinit(
  TARGETS ${EXE_NAME}
  MODULES ${VTK_LIBRARIES}
)
