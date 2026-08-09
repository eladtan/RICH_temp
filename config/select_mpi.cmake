include_guard(GLOBAL)

include(CMakeParseArguments)

function(_rich_find_mpi_sibling output directory)
    set(_found "")
    foreach(_name IN LISTS ARGN)
        if(EXISTS "${directory}/${_name}" AND NOT IS_DIRECTORY "${directory}/${_name}")
            set(_found "${directory}/${_name}")
            break()
        endif()
    endforeach()
    set(${output} "${_found}" PARENT_SCOPE)
endfunction()

function(_rich_identify_mpi launcher output_implementation output_version)
    execute_process(
        COMMAND "${launcher}" --version
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    string(CONCAT _version "${_stdout}" " " "${_stderr}")

    set(_implementation "")
    if(_result EQUAL 0)
        if(_version MATCHES "[Oo]pen.?MPI|OpenRTE|ORTE")
            set(_implementation "OpenMPI")
        elseif(_version MATCHES "Intel|I_MPI")
            set(_implementation "IntelMPI")
        elseif(_version MATCHES "MPICH|HYDRA")
            set(_implementation "MPICH")
        endif()
    endif()

    set(${output_implementation} "${_implementation}" PARENT_SCOPE)
    set(${output_version} "${_version}" PARENT_SCOPE)
endfunction()

function(_rich_mpi_cxx_backend wrapper implementation output_backend output_result)
    if(implementation STREQUAL "OpenMPI")
        set(_show_argument "-showme:command")
    else()
        set(_show_argument "-show")
    endif()

    execute_process(
        COMMAND "${wrapper}" "${_show_argument}"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(_stdout)
        set(_backend "${_stdout}")
    else()
        set(_backend "${_stderr}")
    endif()

    set(${output_backend} "${_backend}" PARENT_SCOPE)
    set(${output_result} "${_result}" PARENT_SCOPE)
endfunction()

function(rich_select_mpi_toolchain)
    set(_options REQUIRED FORTRAN_REQUIRED)
    cmake_parse_arguments(RICH_MPI "${_options}" "" "" ${ARGN})

    if(NOT DEFINED RICH_MPI_IMPLEMENTATION)
        set(RICH_MPI_IMPLEMENTATION "AUTO" CACHE STRING
            "MPI implementation to select: AUTO, OpenMPI, IntelMPI, or MPICH")
    endif()
    set(RICH_MPI_IMPLEMENTATION "${RICH_MPI_IMPLEMENTATION}" CACHE STRING
        "MPI implementation to select: AUTO, OpenMPI, IntelMPI, or MPICH" FORCE)
    set_property(CACHE RICH_MPI_IMPLEMENTATION PROPERTY STRINGS
        AUTO OpenMPI IntelMPI MPICH)
    set(RICH_MPI_WRAPPER_DIR "${RICH_MPI_WRAPPER_DIR}" CACHE PATH
        "Directory containing one coherent MPI wrapper suite")

    string(TOUPPER "${RICH_MPI_IMPLEMENTATION}" _requested_implementation)
    if(NOT _requested_implementation MATCHES "^(AUTO|OPENMPI|INTELMPI|MPICH)$")
        message(FATAL_ERROR
            "Unsupported RICH_MPI_IMPLEMENTATION='${RICH_MPI_IMPLEMENTATION}'. "
            "Expected AUTO, OpenMPI, IntelMPI, or MPICH.")
    endif()

    set(_candidate_directories "")
    if(RICH_MPI_WRAPPER_DIR)
        get_filename_component(_explicit_directory
            "${RICH_MPI_WRAPPER_DIR}" ABSOLUTE)
        list(APPEND _candidate_directories "${_explicit_directory}")
    else()
        file(TO_CMAKE_PATH "$ENV{PATH}" _path_directories)

        # Prefer the standard MPI wrapper suite.  Vendor compiler modules may
        # prepend their own mpiexec/mpicc/mpicxx while leaving mpifort from the
        # MPI module later on PATH.  Anchoring on a complete suite prevents a
        # mixed implementation without relying on module names or hostnames.
        foreach(_directory IN LISTS _path_directories)
            if(EXISTS "${_directory}/mpifort")
                list(APPEND _candidate_directories "${_directory}")
            endif()
        endforeach()

        # Older OpenMPI/MPICH installations may expose only mpif90.  Do not
        # treat Intel MPI's generic mpif90 as the standard-suite anchor; its
        # vendor wrappers are handled explicitly below.
        foreach(_directory IN LISTS _path_directories)
            if(EXISTS "${_directory}/mpif90" AND
               NOT EXISTS "${_directory}/mpiifx" AND
               NOT EXISTS "${_directory}/mpiifort")
                list(APPEND _candidate_directories "${_directory}")
            endif()
        endforeach()

        # Intel MPI uses vendor-specific Fortran wrapper names on some
        # installations.  Keep it as a fallback when no standard suite matches.
        foreach(_directory IN LISTS _path_directories)
            if(EXISTS "${_directory}/mpiifx" OR EXISTS "${_directory}/mpiifort")
                list(APPEND _candidate_directories "${_directory}")
            endif()
        endforeach()

        # C/C++-only MPI installations remain usable when Fortran MPI is not
        # required by the caller.
        if(NOT RICH_MPI_FORTRAN_REQUIRED)
            foreach(_directory IN LISTS _path_directories)
                if(EXISTS "${_directory}/mpicxx" OR EXISTS "${_directory}/mpic++")
                    list(APPEND _candidate_directories "${_directory}")
                endif()
            endforeach()
        endif()
    endif()
    list(REMOVE_DUPLICATES _candidate_directories)

    if(DEFINED INTEL)
        set(_expected_backend_regex "^(icpx|icpc)$")
        set(_expected_backend_description "icpx or icpc")
    elseif(DEFINED GNU)
        set(_expected_backend_regex "^(g\\+\\+|c\\+\\+)$")
        set(_expected_backend_description "g++ or c++")
    else()
        message(FATAL_ERROR "Cannot select MPI for an unknown compiler family")
    endif()

    set(_selection_diagnostics "")
    set(_selected_implementation "")
    foreach(_directory IN LISTS _candidate_directories)
        _rich_find_mpi_sibling(_launcher "${_directory}" mpiexec mpirun)
        if(NOT _launcher)
            list(APPEND _selection_diagnostics
                "${_directory}: no mpiexec or mpirun")
            continue()
        endif()

        _rich_identify_mpi("${_launcher}" _implementation _version)
        if(NOT _implementation)
            list(APPEND _selection_diagnostics
                "${_directory}: launcher '${_launcher}' failed or is unrecognized: ${_version}")
            continue()
        endif()
        string(TOUPPER "${_implementation}" _implementation_upper)
        if(NOT _requested_implementation STREQUAL "AUTO" AND
           NOT _requested_implementation STREQUAL "${_implementation_upper}")
            list(APPEND _selection_diagnostics
                "${_directory}: ${_implementation}, requested ${RICH_MPI_IMPLEMENTATION}")
            continue()
        endif()

        if(_implementation STREQUAL "IntelMPI" AND DEFINED INTEL)
            set(_c_names mpiicx mpiicc mpicc)
            set(_cxx_names mpiicpx mpiicpc mpicxx mpic++)
            set(_fortran_names mpiifx mpiifort mpifort mpif90)
        else()
            set(_c_names mpicc)
            set(_cxx_names mpicxx mpic++ mpiCC)
            set(_fortran_names mpifort mpif90 mpiifx mpiifort)
        endif()

        _rich_find_mpi_sibling(_c_wrapper "${_directory}" ${_c_names})
        _rich_find_mpi_sibling(_cxx_wrapper "${_directory}" ${_cxx_names})
        _rich_find_mpi_sibling(_fortran_wrapper "${_directory}" ${_fortran_names})
        if(NOT _c_wrapper OR NOT _cxx_wrapper OR
           (RICH_MPI_FORTRAN_REQUIRED AND NOT _fortran_wrapper))
            list(APPEND _selection_diagnostics
                "${_directory}: incomplete ${_implementation} wrapper suite")
            continue()
        endif()

        _rich_mpi_cxx_backend("${_cxx_wrapper}" "${_implementation}"
            _backend _backend_result)
        if(NOT _backend_result EQUAL 0)
            list(APPEND _selection_diagnostics
                "${_directory}: could not query '${_cxx_wrapper}' backend")
            continue()
        endif()
        separate_arguments(_backend_words UNIX_COMMAND "${_backend}")
        if(NOT _backend_words)
            list(APPEND _selection_diagnostics
                "${_directory}: '${_cxx_wrapper}' returned no backend command")
            continue()
        endif()
        list(GET _backend_words 0 _backend_executable)
        get_filename_component(_backend_name "${_backend_executable}" NAME)
        if(NOT _backend_name MATCHES "${_expected_backend_regex}")
            list(APPEND _selection_diagnostics
                "${_directory}: ${_implementation} uses '${_backend_name}', "
                "expected ${_expected_backend_description}")
            continue()
        endif()

        set(_selected_directory "${_directory}")
        set(_selected_launcher "${_launcher}")
        set(_selected_c_wrapper "${_c_wrapper}")
        set(_selected_cxx_wrapper "${_cxx_wrapper}")
        set(_selected_fortran_wrapper "${_fortran_wrapper}")
        set(_selected_implementation "${_implementation}")
        set(_selected_backend "${_backend}")
        break()
    endforeach()

    if(NOT _selected_implementation)
        string(JOIN "\n  " _diagnostic_text ${_selection_diagnostics})
        string(CONCAT _failure_message
            "No coherent MPI wrapper suite matches ${CONFIG}.\n"
            "  ${_diagnostic_text}\n"
            "Set RICH_MPI_WRAPPER_DIR to the desired suite's bin directory, "
            "or set RICH_MPI_IMPLEMENTATION explicitly.")
        if(RICH_MPI_REQUIRED)
            message(FATAL_ERROR "${_failure_message}")
        else()
            message(STATUS "${_failure_message}")
            return()
        endif()
    endif()

    # FindMPI stores derived include flags, libraries, and probe results in the
    # cache.  All of them must be refreshed together; retaining only
    # MPI_*_COMPILE_OPTIONS is enough to combine Intel MPI headers with
    # OpenMPI libraries.
    get_cmake_property(_cache_variables CACHE_VARIABLES)
    foreach(_cache_variable IN LISTS _cache_variables)
        if(_cache_variable MATCHES "^MPIEXEC_" OR
           _cache_variable MATCHES "^MPI_")
            unset("${_cache_variable}" CACHE)
        endif()
    endforeach()

    set(MPIEXEC_PATH "${_selected_launcher}" CACHE FILEPATH
        "MPI launcher from the selected coherent wrapper suite" FORCE)
    set(MPIEXEC_EXECUTABLE "${_selected_launcher}" CACHE FILEPATH
        "MPI launcher from the selected coherent wrapper suite" FORCE)
    set(MPI_C_COMPILER "${_selected_c_wrapper}" CACHE FILEPATH
        "MPI C wrapper from the selected coherent wrapper suite" FORCE)
    set(MPI_CXX_COMPILER "${_selected_cxx_wrapper}" CACHE FILEPATH
        "MPI C++ wrapper from the selected coherent wrapper suite" FORCE)
    if(_selected_fortran_wrapper)
        set(MPI_Fortran_COMPILER "${_selected_fortran_wrapper}" CACHE FILEPATH
            "MPI Fortran wrapper from the selected coherent wrapper suite" FORCE)
    endif()
    set(RICH_SELECTED_MPI_IMPLEMENTATION "${_selected_implementation}"
        CACHE INTERNAL "Selected MPI implementation" FORCE)
    set(RICH_SELECTED_MPI_WRAPPER_DIR "${_selected_directory}"
        CACHE INTERNAL "Selected MPI wrapper directory" FORCE)

    set(MPI_IMPL "${_selected_implementation}" PARENT_SCOPE)
    message(STATUS "MPI implementation: ${_selected_implementation}")
    message(STATUS "MPI wrapper directory: ${_selected_directory}")
    message(STATUS "MPI launcher: ${_selected_launcher}")
    message(STATUS "MPI C wrapper: ${_selected_c_wrapper}")
    message(STATUS "MPI CXX wrapper: ${_selected_cxx_wrapper}")
    message(STATUS "MPI Fortran wrapper: ${_selected_fortran_wrapper}")
    message(STATUS "MPI CXX backend: ${_selected_backend}")
endfunction()

function(rich_validate_mpi_toolchain)
    if(NOT TARGET MPI::MPI_CXX)
        message(FATAL_ERROR "MPI::MPI_CXX is unavailable for MPI validation")
    endif()

    if(RICH_SELECTED_MPI_IMPLEMENTATION STREQUAL "OpenMPI")
        set(_implementation_guard
            "#ifndef OPEN_MPI\n#error MPI header is not OpenMPI\n#endif")
    elseif(RICH_SELECTED_MPI_IMPLEMENTATION STREQUAL "IntelMPI")
        set(_implementation_guard
            "#ifndef I_MPI_VERSION\n#error MPI header is not Intel MPI\n#endif")
    elseif(RICH_SELECTED_MPI_IMPLEMENTATION STREQUAL "MPICH")
        set(_implementation_guard
            "#if !defined(MPICH) && !defined(MPICH_NAME) && !defined(MPICH_VERSION)\n#error MPI header is not MPICH\n#endif")
    else()
        message(FATAL_ERROR "No selected MPI implementation is available for validation")
    endif()

    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
    set(CMAKE_REQUIRED_QUIET TRUE)
    unset(RICH_MPI_HEADER_LIBRARY_COHERENT CACHE)
    check_cxx_source_compiles(
        "#include <mpi.h>\n${_implementation_guard}\nint main() {\n  MPI_Comm graph = MPI_COMM_NULL;\n  return MPI_Dist_graph_create_adjacent(\n      MPI_COMM_WORLD, 0, nullptr, MPI_UNWEIGHTED,\n      0, nullptr, MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &graph);\n}\n"
        RICH_MPI_HEADER_LIBRARY_COHERENT
    )
    if(NOT RICH_MPI_HEADER_LIBRARY_COHERENT)
        message(FATAL_ERROR
            "MPI headers and libraries are incoherent. Selected "
            "${RICH_SELECTED_MPI_IMPLEMENTATION} wrappers from "
            "${RICH_SELECTED_MPI_WRAPPER_DIR}, but FindMPI produced:\n"
            "  MPI_CXX_INCLUDE_DIRS=${MPI_CXX_INCLUDE_DIRS}\n"
            "  MPI_CXX_COMPILE_OPTIONS=${MPI_CXX_COMPILE_OPTIONS}\n"
            "  MPI_CXX_LIBRARIES=${MPI_CXX_LIBRARIES}\n"
            "Remove the stale build cache or select a suite with "
            "RICH_MPI_WRAPPER_DIR.")
    endif()
    message(STATUS "MPI header/library coherence: verified")
endfunction()
