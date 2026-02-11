set(path_env $ENV{PATH})

# Split PATH into a CMake list
string(REPLACE ":" ";" path_env_list "${path_env}")

# TODO: change if necessary
set(VTUNE_DIRECTORY "/software/x86_64/5.14.0/Intel/OneApi/2024.2.1/vtune/latest/")

if(NOT VTUNE_DIRECTORY)
    # Extract likely VTUNE prefixes
    set(VTUNE_DIRECTORY "")
    foreach(dir IN LISTS path_env_list)
        if(dir MATCHES "/vtune")
            string(REGEX REPLACE "/bin(32|64)?$" "" prefix "${dir}")
            # if there is already a prefix - error, multiple VTUNE installations
            if(VTUNE_DIRECTORY)
                message(FATAL_ERROR "Multiple VTUNE installations found in PATH: ${VTUNE_DIRECTORY} and ${prefix}")
            endif()
            message(STATUS "Found VTUNE installation at: ${prefix}")
            # Add the prefix to the list
            message(STATUS "Using vtune: ${prefix}")
            list(APPEND VTUNE_DIRECTORY "${prefix}")
        endif()
    endforeach()

    # if no VTUNE_DIRECTORY found, error
    if(NOT VTUNE_DIRECTORY)
        message("No VTUNE installation found in PATH environment variable")
    endif()
endif()

if(VTUNE_DIRECTORY)
    set(VTUNE_INCLUDE "${VTUNE_DIRECTORY}/include")
    set(VTUNE_LIB_DIRECTORY "${VTUNE_DIRECTORY}/lib64")
endif()
