# ==================== Locate VTK ====================

set(ld_lib_path $ENV{LD_LIBRARY_PATH})
string(REPLACE ":" ";" ld_lib_path_list "${ld_lib_path}")

if(NOT VTK_DIRECTORY)
    set(_vtk_prefixes "")
    foreach(dir IN LISTS ld_lib_path_list)
        if(dir MATCHES "/vtk" OR dir MATCHES "/VTK")
            string(REGEX REPLACE "/lib(64)?$" "" prefix "${dir}")
            list(FIND _vtk_prefixes "${prefix}" _idx)
            if(_idx EQUAL -1)
                list(APPEND _vtk_prefixes "${prefix}")
                message(STATUS "Found VTK candidate: ${prefix}")
            endif()
        endif()
    endforeach()

    list(LENGTH _vtk_prefixes _vtk_count)

    if(_vtk_count EQUAL 0)
        message(FATAL_ERROR "No VTK installation found in LD_LIBRARY_PATH")
    elseif(_vtk_count GREATER 1)
        message(FATAL_ERROR "Multiple distinct VTK installations found in LD_LIBRARY_PATH: ${_vtk_prefixes}")
    endif()

    list(GET _vtk_prefixes 0 VTK_DIRECTORY)

    set(VTK_LIB_DIRECTORY "${VTK_DIRECTORY}/lib64")
    set(VTK_INCLUDE "${VTK_DIRECTORY}/include")
    message(STATUS "Using VTK: ${VTK_DIRECTORY}")
endif()

# ==================== Find VTK package ====================

# Some VTK component graphs depend on MPI even for a serial RICH build.  Seed
# that transitive FindMPI call with one coherent wrapper suite.
if(NOT DEFINED MPI)
    include(${CMAKE_CURRENT_LIST_DIR}/select_mpi.cmake)
    rich_select_mpi_toolchain()
endif()

set(VTK_COMPONENTS
    CommonCore
    CommonColor
    CommonDataModel
    CommonTransforms
    FiltersGeneral
    FiltersSources
    IOXML
    InteractionStyle
)
if(DEFINED MPI)
    list(APPEND VTK_COMPONENTS IOParallelXML ParallelMPI)
endif()

if(DEFINED MPI)
    find_package(VTK COMPONENTS ${VTK_COMPONENTS}
        NO_MODULE
        PATHS ${VTK_DIRECTORY} NO_DEFAULT_PATH
    )
else()
    # QUIET suppresses "Found MPI" messages from VTK's internal transitive
    # dependencies (e.g. FiltersGeneral -> ParallelDIY -> mpi)
    find_package(VTK QUIET COMPONENTS ${VTK_COMPONENTS}
        NO_MODULE
        PATHS ${VTK_DIRECTORY} NO_DEFAULT_PATH
    )
endif()

if(NOT VTK_FOUND)
    message(FATAL_ERROR "VTK not found in ${VTK_DIRECTORY}")
endif()

# ==================== VTK-MPI compatibility ====================

if(DEFINED MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    rich_validate_mpi_toolchain()

    # VTK::ParallelMPI is a shared imported target. Inspect its direct MPI
    # dependency: a C++ probe based on vtkMPICommunicatorOpaqueComm is not
    # reliable because VTK intentionally exports vendor-neutral MPI compile
    # definitions, which can change that private wrapper's apparent signature
    # in an isolated try_compile.  The shared-library dependency is the actual
    # MPI implementation that VTK will load at runtime.
    if(MPI_IMPL STREQUAL "OpenMPI")
        set(_vtk_parallel_mpi_library "")
        foreach(_vtk_imported_config "" "_RELEASE" "_RELWITHDEBINFO" "_DEBUG")
            get_target_property(_vtk_candidate VTK::ParallelMPI "IMPORTED_LOCATION${_vtk_imported_config}")
            if(_vtk_candidate AND EXISTS "${_vtk_candidate}")
                set(_vtk_parallel_mpi_library "${_vtk_candidate}")
                break()
            endif()
        endforeach()

        if(NOT _vtk_parallel_mpi_library)
            message(FATAL_ERROR
                "Could not resolve VTK::ParallelMPI's shared library for OpenMPI validation")
        endif()

        # Prefer the binary inspector supplied by CMake's active toolchain.
        # Fall back to a PATH lookup, but do not make an ELF-only utility a
        # prerequisite on platforms where this compatibility check is not
        # available.
        set(_rich_readelf "${CMAKE_READELF}")
        if(NOT _rich_readelf)
            find_program(_rich_readelf NAMES readelf llvm-readelf)
        endif()
        if(_rich_readelf)
            execute_process(
                COMMAND "${_rich_readelf}" -d "${_vtk_parallel_mpi_library}"
                OUTPUT_VARIABLE _vtk_dynamic_dependencies
                RESULT_VARIABLE _vtk_readelf_result
                ERROR_VARIABLE _vtk_readelf_error
            )
            if(NOT _vtk_readelf_result EQUAL 0)
                message(FATAL_ERROR
                    "Could not inspect ${_vtk_parallel_mpi_library}: ${_vtk_readelf_error}")
            endif()
            if(_vtk_dynamic_dependencies MATCHES "libmpi\\.so\\.12")
                message(FATAL_ERROR
                    "VTK::ParallelMPI at ${_vtk_parallel_mpi_library} requires "
                    "libmpi.so.12, which is incompatible with the selected OpenMPI toolchain.")
            endif()
        else()
            message(WARNING
                "No readelf-compatible binary inspector is available; skipping "
                "the optional VTK/OpenMPI shared-library dependency check")
        endif()
    endif()
else()
    # VTK built with MPI declares internal dependency chains
    # (e.g. FiltersGeneral -> ParallelDIY -> mpi) that pull libmpi
    # into the link even when only non-MPI components are requested.
    # Strip those for serial builds.
    foreach(_tgt VTK::mpi VTK::ParallelMPI VTK::ParallelCore VTK::ParallelDIY)
        if(TARGET ${_tgt})
            set_target_properties(${_tgt} PROPERTIES
                INTERFACE_LINK_LIBRARIES ""
                INTERFACE_INCLUDE_DIRECTORIES ""
            )
        endif()
    endforeach()
endif()
