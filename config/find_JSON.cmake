set(ld_lib_path $ENV{LD_LIBRARY_PATH})

# Split LD_LIBRARY_PATH into a CMake list
string(REPLACE ":" ";" ld_lib_path_list "${ld_lib_path}")

if(NOT JSON_DIRECTORY)
    # Extract likely JSON prefixes
    set(JSON_DIRECTORY "")
    foreach(dir IN LISTS ld_lib_path_list)
        if(dir MATCHES "/jsoncpp")
            string(REGEX REPLACE "/lib(64)?$" "" prefix "${dir}")
        # if there is already a prefix - error, multiple JSON installations
            if(JSON_DIRECTORY)
                message(FATAL_ERROR "Multiple JSON installations found in LD_LIBRARY_PATH: ${JSON_DIRECTORY} and ${prefix}")
            endif()
            message(STATUS "Using json cpp: ${prefix}")
            # Add the prefix to the list
            list(APPEND JSON_DIRECTORY "${prefix}")
        endif()
    endforeach()

    # if no JSON_DIRECTORY found, message that it compiles with no json
    if(NOT JSON_DIRECTORY)
        message("No JSON installation found in LD_LIBRARY_PATH environment variable")
    else()
        set(JSONCPP_INCLUDE "${JSON_DIRECTORY}/include")
        set(JSONCPP_LIB_DIRECTORY "${JSON_DIRECTORY}/lib64")
    endif()
endif()