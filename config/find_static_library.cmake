function(find_static_library OUT_VAR LIBNAME)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs PATHS)
    cmake_parse_arguments(FSL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Backup old suffixes
    set(_old_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")

    if(WIN32)
        set(CMAKE_FIND_LIBRARY_SUFFIXES .lib)
    else()
        set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    endif()

    find_library(${OUT_VAR}
        NAMES ${LIBNAME}
        PATHS ${FSL_PATHS}
    )

    # Restore suffixes
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${_old_suffixes}")

    if(NOT ${OUT_VAR})
        message(FATAL_ERROR "Static library ${LIBNAME} not found in paths: ${FSL_PATHS}")
    endif()
endfunction()
