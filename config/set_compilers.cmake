set(GNU_CC_NAME "gcc")
set(GNU_CXX_NAME "g++")
set(GNU_Fortran_NAME "gfortran")
set(INTEL_CC_NAME "icx")
set(INTEL_CXX_NAME "icpx")
set(INTEL_Fortran_NAME "ifx")

function(find_executable_path exe_name out_var)
    string(TOUPPER "${exe_name}" _var_suffix)
    set(_cache_var "FIND_EXECUTABLE_PATH_${_var_suffix}_PATH")

    find_program(${_cache_var} NAMES ${exe_name})
    if(${_cache_var})
        set(${out_var} "${${_cache_var}}" PARENT_SCOPE)
        message(STATUS "Found executable '${exe_name}' at: ${${_cache_var}}")
    else()
        message(FATAL_ERROR "Executable '${exe_name}' not found")
    endif()
endfunction()

if(DEFINED GNU)
    find_executable_path(${GNU_CC_NAME} CMAKE_C_COMPILER)
    find_executable_path(${GNU_CXX_NAME} CMAKE_CXX_COMPILER)
    find_executable_path(${GNU_Fortran_NAME} CMAKE_Fortran_COMPILER)
elseif(DEFINED INTEL)
    find_executable_path(${INTEL_CC_NAME} CMAKE_C_COMPILER)
    find_executable_path(${INTEL_CXX_NAME} CMAKE_CXX_COMPILER)
    find_executable_path(${INTEL_Fortran_NAME} CMAKE_Fortran_COMPILER)
endif()

message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Fortran Compiler: ${CMAKE_Fortran_COMPILER}")

if(DEFINED MPI)
    # Detect MPI implementation from mpiexec
    find_program(MPIEXEC_PATH mpiexec REQUIRED)
    execute_process(
        COMMAND ${MPIEXEC_PATH} --version
        OUTPUT_VARIABLE mpi_version_out
        ERROR_VARIABLE mpi_version_err
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    string(CONCAT mpi_version "${mpi_version_out}" " " "${mpi_version_err}")

    if(mpi_version MATCHES "[Oo]pen.?MPI|OpenRTE|ORTE")
        set(MPI_IMPL "OpenMPI")
    elseif(mpi_version MATCHES "Intel|I_MPI")
        set(MPI_IMPL "IntelMPI")
    elseif(mpi_version MATCHES "MPICH|HYDRA")
        set(MPI_IMPL "MPICH")
    else()
        message(FATAL_ERROR
            "Unrecognized MPI implementation from 'mpiexec --version':\n${mpi_version}\n"
            "Supported implementations: OpenMPI, IntelMPI, MPICH")
    endif()

    message(STATUS "MPI implementation: ${MPI_IMPL}")

    # Find MPI wrappers from PATH, skipping CMake-internal paths that
    # Intel OneAPI pollutes. Cache MPI_CXX_COMPILER / MPI_C_COMPILER so
    # that dependencies calling find_package(MPI) internally (e.g. VTK's
    # ParallelMPI) also find the correct MPI.
    find_program(MPI_CXX_COMPILER NAMES mpicxx mpic++ NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH REQUIRED)
    find_program(MPI_C_COMPILER   NAMES mpicc         NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH REQUIRED)

    # Select the Fortran wrapper from the same MPI implementation.  FindMPI
    # otherwise sees the Intel Fortran compiler and can independently select
    # Intel MPI even when C and CXX use OpenMPI.
    if(MPI_IMPL STREQUAL "OpenMPI")
        set(_mpi_fortran_wrappers mpifort mpif90)
    elseif(MPI_IMPL STREQUAL "IntelMPI")
        set(_mpi_fortran_wrappers mpiifx mpiifort)
    else()
        set(_mpi_fortran_wrappers mpifort mpif90)
    endif()
    unset(MPI_Fortran_COMPILER CACHE)
    unset(MPI_Fortran_COMPILER)
    find_program(MPI_Fortran_COMPILER
        NAMES ${_mpi_fortran_wrappers}
        NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH REQUIRED
    )
    message(STATUS "MPI Fortran wrapper: ${MPI_Fortran_COMPILER}")

    # OneAPI exports its own MPI headers from the compiler environment.  When
    # OpenMPI is selected, compiling directly with icx/icpx/ifx can therefore
    # pick up Intel MPI's mpi.h even though linking uses OpenMPI.  Configure
    # through OpenMPI's wrappers instead: they still invoke the Intel
    # compilers, but add OpenMPI's headers before the OneAPI defaults.
    if(DEFINED INTEL AND MPI_IMPL STREQUAL "OpenMPI")
        execute_process(
            COMMAND ${MPI_CXX_COMPILER} -showme:command
            OUTPUT_VARIABLE _openmpi_cxx_backend
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _openmpi_cxx_backend_result
        )
        if(NOT _openmpi_cxx_backend_result EQUAL 0 OR
           NOT _openmpi_cxx_backend MATCHES "(^|.*/)icpx( |$)")
            message(FATAL_ERROR
                "intelReleaseMPI with OpenMPI requires mpicxx to invoke icpx; "
                "got '${_openmpi_cxx_backend}' from ${MPI_CXX_COMPILER}")
        endif()

        set(CMAKE_C_COMPILER       "${MPI_C_COMPILER}"       CACHE FILEPATH "OpenMPI C wrapper" FORCE)
        set(CMAKE_CXX_COMPILER     "${MPI_CXX_COMPILER}"     CACHE FILEPATH "OpenMPI C++ wrapper" FORCE)
        set(CMAKE_Fortran_COMPILER "${MPI_Fortran_COMPILER}" CACHE FILEPATH "OpenMPI Fortran wrapper" FORCE)
        message(STATUS "Intel/OpenMPI compiler wrappers: ${CMAKE_C_COMPILER}; ${CMAKE_CXX_COMPILER}; ${CMAKE_Fortran_COMPILER}")
    endif()

    # Query compile and link flags from the MPI wrapper
    if(MPI_IMPL STREQUAL "OpenMPI")
        execute_process(COMMAND ${MPI_CXX_COMPILER} -showme:compile
            OUTPUT_VARIABLE MPI_COMPILE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${MPI_CXX_COMPILER} -showme:link
            OUTPUT_VARIABLE MPI_LINK_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    else()
        # IntelMPI and MPICH both support -show
        execute_process(COMMAND ${MPI_CXX_COMPILER} -show
            OUTPUT_VARIABLE mpi_show OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_VARIABLE mpi_show_err ERROR_STRIP_TRAILING_WHITESPACE)
        message(STATUS "mpicxx -show raw:  [${mpi_show}]")
        if(mpi_show_err)
            message(STATUS "mpicxx -show err:  [${mpi_show_err}]")
        endif()
        # Strip the compiler name (first word)
        string(REGEX REPLACE "^[^ ]+ +" "" mpi_all_flags "${mpi_show}")
        # Compile flags: -I and -D entries
        string(REGEX MATCHALL "-[ID][^ ]+" _compile_list "${mpi_all_flags}")
        string(REPLACE ";" " " MPI_COMPILE_FLAGS "${_compile_list}")
        # Link flags: the full set
        set(MPI_LINK_FLAGS "${mpi_all_flags}")

        # Fallback: if -show didn't expose include/library paths, derive from
        # the wrapper's installation directory (common with Intel MPI)
        get_filename_component(_mpi_bin_dir "${MPI_CXX_COMPILER}" DIRECTORY)
        get_filename_component(_mpi_root "${_mpi_bin_dir}" DIRECTORY)
        message(STATUS "MPI root (derived): ${_mpi_root}")

        if(NOT MPI_COMPILE_FLAGS MATCHES "-I")
            if(EXISTS "${_mpi_root}/include/mpi.h")
                set(MPI_COMPILE_FLAGS "-I${_mpi_root}/include")
            else()
                message(WARNING "MPI fallback: ${_mpi_root}/include/mpi.h not found")
            endif()
        endif()

        if(NOT MPI_LINK_FLAGS MATCHES "-lmpi")
            if(EXISTS "${_mpi_root}/lib/release")
                set(MPI_LINK_FLAGS "-L${_mpi_root}/lib/release -L${_mpi_root}/lib -lmpi ${MPI_LINK_FLAGS}")
            elseif(EXISTS "${_mpi_root}/lib")
                set(MPI_LINK_FLAGS "-L${_mpi_root}/lib -lmpi ${MPI_LINK_FLAGS}")
            elseif(EXISTS "${_mpi_root}/lib64")
                set(MPI_LINK_FLAGS "-L${_mpi_root}/lib64 -lmpi ${MPI_LINK_FLAGS}")
            else()
                message(WARNING "MPI fallback: no lib directory found under ${_mpi_root}")
            endif()
        endif()
    endif()

    message(STATUS "MPI wrapper:       ${MPI_CXX_COMPILER}")
    message(STATUS "MPI compile flags: ${MPI_COMPILE_FLAGS}")
    message(STATUS "MPI link flags:    ${MPI_LINK_FLAGS}")

endif()
