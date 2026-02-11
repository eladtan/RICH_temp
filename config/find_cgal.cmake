set(path_env $ENV{PATH})

# Split PATH into a CMake list
string(REPLACE ":" ";" path_env_list "${path_env}")

if(NOT CGAL_DIRECTORY)
    # Extract likely CGAL prefixes
    set(CGAL_DIRECTORY "")
    foreach(dir IN LISTS path_env_list)
        if(dir MATCHES "/cgal")
            string(REGEX REPLACE "/bin" "" prefix "${dir}")
            # if there is already a prefix - error, multiple CGAL installations
            if(CGAL_DIRECTORY)
                message(FATAL_ERROR "Multiple CGAL installations found in PATH: ${CGAL_DIRECTORY} and ${prefix}")
            endif()
            message("Found CGAL installation at: ${prefix}")
            # Add the prefix to the list
            list(APPEND CGAL_DIRECTORY "${prefix}")
        endif()
    endforeach()

    # if no CGAL_DIRECTORY found, error
    if(NOT CGAL_DIRECTORY)
        message("No CGAL installation found in PATH environment variable")
    endif()
endif()