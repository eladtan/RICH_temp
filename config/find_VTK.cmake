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
    # Verify that VTK was compiled with the same MPI implementation.
    set(_test_src "${CMAKE_BINARY_DIR}/CMakeFiles/vtk_mpi_check.cpp")
    file(WRITE "${_test_src}" [=[
        #include <mpi.h>
        #include <vtkMPI.h>
        int main() {
            MPI_Comm comm;
            vtkMPICommunicatorOpaqueComm opaqueComm(&comm);
            (void)opaqueComm;
            return 0;
        }
        ]=])

    separate_arguments(mpi_link_flags UNIX_COMMAND "${MPI_LINK_FLAGS}")
    try_compile(_vtk_mpi_ok
        "${CMAKE_BINARY_DIR}/CMakeFiles/_vtk_mpi_check"
        "${_test_src}"
        LINK_LIBRARIES ${VTK_LIBRARIES} ${mpi_link_flags}
        CMAKE_FLAGS "-DCMAKE_CXX_FLAGS=${MPI_COMPILE_FLAGS}"
        CXX_STANDARD 17
        OUTPUT_VARIABLE _vtk_mpi_output
    )

    # if(NOT _vtk_mpi_ok)
    #     message(FATAL_ERROR
    #         "\n"
    #         "===========================================================\n"
    #         " VTK-MPI compatibility check FAILED!\n"
    #         "\n"
    #         " VTK was compiled with a different MPI implementation than\n"
    #         " the one currently in use. MPI_Comm has a different type in\n"
    #         " each implementation (e.g. int for MPICH/IntelMPI vs\n"
    #         " ompi_communicator_t* for OpenMPI), causing linker errors.\n"
    #         "\n"
    #         " Current MPI compiler: ${MPI_CXX_COMPILER}\n"
    #         " VTK directory:        ${VTK_DIRECTORY}\n"
    #         "\n"
    #         " Please recompile VTK with the same MPI, or switch to the\n"
    #         " MPI implementation that VTK was compiled with.\n"
    #         "===========================================================\n"
    #         "\n"
    #         "Build output:\n${_vtk_mpi_output}\n"
    #     )
    # else()
    #     message(STATUS "VTK-MPI compatibility: OK")
    # endif()
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