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

message(STATUS "CXX Standard: " ${CMAKE_CXX_STANDARD})
message(STATUS "Build Type: " ${CMAKE_BUILD_TYPE})

message(STATUS "CMAKE_CXX_FLAGS = " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_CXX_FLAGS_RELEASE = " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "CMAKE_CXX_FLAGS_DEBUG = " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "CMAKE_Fortran_FLAGS = " ${CMAKE_Fortran_FLAGS})
message(STATUS "CMAKE_Fortran_FLAGS_RELEASE = " ${CMAKE_Fortran_FLAGS_RELEASE})
message(STATUS "CMAKE_Fortran_FLAGS_DEBUG = " ${CMAKE_Fortran_FLAGS_DEBUG})

project(rich)

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

# message(STATUS "Adding linker -flto option")
# add_link_options(-flto)

# files to compile (source library)
file(GLOB_RECURSE SRC FOLLOW_SYMLINKS "${PROJECT_SOURCE_DIR}/*.cpp" "${PROJECT_SOURCE_DIR}/*.c")

if(TEST_DIR)
    if(EXISTS "${PROJECT_SOURCE_DIR}/../runs/${TEST_DIR}")
        set(TEST_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../runs/${TEST_DIR}")
    elseif(EXISTS "${PROJECT_SOURCE_DIR}/../${TEST_DIR}")
        set(TEST_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../${TEST_DIR}")
    else()
        message(FATAL_ERROR "Could not find test directory. Checked: ${PROJECT_SOURCE_DIR}/../runs/${TEST_DIR} and ${PROJECT_SOURCE_DIR}/../${TEST_DIR}")
    endif()
    file(GLOB_RECURSE TEST_SRC FOLLOW_SYMLINKS "${TEST_SOURCE_DIR}/*.cpp" "${TEST_SOURCE_DIR}/*.c")
endif()

include(${CMAKE_SOURCE_DIR}/../config/find_HDF5.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_VTK.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_JSON.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_Boost.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_cgal.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_python.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_vtune.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_ParMETIS.cmake)
include(${CMAKE_SOURCE_DIR}/../config/include_3rdparty.cmake)
include(${CMAKE_SOURCE_DIR}/../config/find_pybind11.cmake)
include(${CMAKE_SOURCE_DIR}/../config/set_placeholders.cmake)

if(DEFINED PYBIND11_DIRECTORY)
    set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION})
    find_package(pybind11 QUIET REQUIRED PATHS ${PYBIND11_DIRECTORY} NO_DEFAULT_PATH)

    if(pybind11_FOUND)
        message(STATUS "pybind11 directory: " ${pybind11_DIR})
        include_directories("${PYBIND11_DIR}/include")
    else()
        message(FATAL_ERROR "PYBIND11 NOT FOUND")
    endif()
else()
    message(STATUS "No pybind11")
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

    # MPI (parallel HDF5 references MPI symbols)
    if(DEFINED MPI)
        find_package(MPI REQUIRED)
        list(APPEND HDF5_LIBS MPI::MPI_C)
    endif()
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
include_directories("${PROJECT_SOURCE_DIR}/..") # TODO: REMOVE!
include_directories("${BOOST_DIR}/include")
include_directories(${HDF5_INCLUDE})

# python
if(PYTHON_INCLUDE AND PYTHON_LIB_DIRECTORY)
    include_directories(${PYTHON_INCLUDE})
    find_library(PYTHON_LIB NAMES ${PYTHON_LIB_NAME} PATHS ${PYTHON_LIB_DIRECTORY} NO_DEFAULT_PATH)
else()
    message(STATUS "NO PYTHON")
endif()

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
    find_static_library(PARMETIS_LIB parmetis PATHS ${PARMETIS_LIB_DIRECTORY})
endif()

# metis
if(METIS_INCLUDE AND METIS_LIB_DIRECTORY)
    include_directories(${METIS_INCLUDE})
    find_library(METIS_LIB metis PATHS ${METIS_LIB_DIRECTORY})
endif()

# gklib
if(GKLIB_INCLUDE AND GKLIB_LIB_DIRECTORY)
    include_directories(${GKLIB_INCLUDE})
    find_static_library(GKLIB_LIB GKlib PATHS ${GKLIB_LIB_DIRECTORY})
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

# ======================= rich_core LIBRARY =======================
add_library(rich_core STATIC ${SRC})
target_include_directories(rich_core PUBLIC "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/..")

find_package(ZLIB REQUIRED)
target_link_libraries(rich_core PUBLIC ${HDF5_LIBS} ${VTK_LIBRARIES} ZLIB::ZLIB)

if(DEFINED JSONCPP_LIB)
    target_link_libraries(rich_core PUBLIC ${JSONCPP_LIB})
endif()

if(DEFINED PARMETIS_LIB)
    target_link_libraries(rich_core PUBLIC ${PARMETIS_LIB})
    add_compile_definitions("WITH_PARMETIS")
endif()

if(DEFINED METIS_LIB)
    target_link_libraries(rich_core PUBLIC ${METIS_LIB})
endif()

if(DEFINED GKLIB_LIB)
    target_link_libraries(rich_core PUBLIC ${GKLIB_LIB})
endif()

if(DEFINED BOOST_DIR)
    target_link_directories(rich_core PUBLIC ${BOOST_DIR}/lib)
endif()

if(pybind11_FOUND)
    target_link_libraries(rich_core PUBLIC pybind11::embed pybind11::module)
endif()

if(DEFINED PYTHON_LIB_LIBRARY)
    target_link_libraries(rich_core PUBLIC ${PYTHON_LIB})
endif()

if((DEFINED VTUNE_INCLUDE) AND (DEFINED VTUNE_LIB_DIRECTORY))
    set(VTUNE_LIB ${VTUNE_LIB_DIRECTORY}/libittnotify.a)
    target_link_libraries(rich_core PUBLIC "${VTUNE_LIB}")
    add_compile_definitions("WITH_VTUNE")
endif()

vtk_module_autoinit(
  TARGETS rich_core
  MODULES ${VTK_LIBRARIES}
)

# ======================= MAIN EXECUTABLE (requires TEST_DIR) =======================
if(TEST_DIR)
    add_executable(${EXE_NAME} ${TEST_SRC})
    target_link_libraries(${EXE_NAME} PRIVATE rich_core)

    if(${MIXED_BUILD})
        message(STATUS "Mixed build: marking some files for debug")
        mark_debug_sources("${EXE_NAME}")
    endif()

    if(CGAL_DIRECTORY)
        # TODO: at least this line
        # target_link_libraries(${EXE_NAME} PRIVATE CGAL::CGAL)
    endif()

    vtk_module_autoinit(
      TARGETS ${EXE_NAME}
      MODULES ${VTK_LIBRARIES}
    )
endif()

# ======================= UNIT TESTS =======================
add_subdirectory(../tests_catch2 tests_catch2)
