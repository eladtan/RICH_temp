set(path_env $ENV{PATH})

# Split PATH into a CMake list
string(REPLACE ":" ";" path_env_list "${path_env}")

if(NOT CGAL_DIRECTORY)
    set(_cgal_prefixes "")
    foreach(dir IN LISTS path_env_list)
        if(dir MATCHES "/cgal")
            string(REGEX REPLACE "/bin" "" prefix "${dir}")
            list(FIND _cgal_prefixes "${prefix}" _idx)
            if(_idx EQUAL -1)
                list(APPEND _cgal_prefixes "${prefix}")
                message(STATUS "Found CGAL candidate: ${prefix}")
            endif()
        endif()
    endforeach()

    list(LENGTH _cgal_prefixes _cgal_count)

    if(_cgal_count EQUAL 0)
        message("No CGAL installation found in PATH environment variable")
    elseif(_cgal_count GREATER 1)
        message(FATAL_ERROR "Multiple distinct CGAL installations found in PATH: ${_cgal_prefixes}")
    else()
        list(GET _cgal_prefixes 0 CGAL_DIRECTORY)
        message(STATUS "Using CGAL: ${CGAL_DIRECTORY}")
    endif()
endif()