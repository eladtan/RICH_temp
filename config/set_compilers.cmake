set(GNU_CC_NAME "gcc")
set(GNU_CXX_NAME "g++")
set(GNU_Fortran_NAME "gfortran")
set(INTEL_CC_NAME "icx")
set(INTEL_CXX_NAME "icpx")
set(INTEL_Fortran_NAME "ifx")

include(${CMAKE_CURRENT_LIST_DIR}/select_mpi.cmake)

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
    rich_select_mpi_toolchain(REQUIRED FORTRAN_REQUIRED)
    set(CMAKE_C_COMPILER "${MPI_C_COMPILER}" CACHE FILEPATH
        "C compiler from the selected MPI suite" FORCE)
    set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}" CACHE FILEPATH
        "C++ compiler from the selected MPI suite" FORCE)
    set(CMAKE_Fortran_COMPILER "${MPI_Fortran_COMPILER}" CACHE FILEPATH
        "Fortran compiler from the selected MPI suite" FORCE)
endif()
