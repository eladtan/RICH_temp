set(ld_lib_path $ENV{LD_LIBRARY_PATH})

# Split LD_LIBRARY_PATH into a CMake list
string(REPLACE ":" ";" ld_lib_path_list "${ld_lib_path}")

# Extract likely VTK prefixes
if(NOT VTK_DIRECTORY)
    set(VTK_DIRECTORY "")
    foreach(dir IN LISTS ld_lib_path_list)
        if(dir MATCHES "/vtk" OR dir MATCHES "/VTK")
            string(REGEX REPLACE "/lib(64)?$" "" prefix "${dir}")
            # if there is already a prefix - error, multiple VTK installations
            if(VTK_DIRECTORY)
                message(FATAL_ERROR "Multiple VTK installations found in LD_LIBRARY_PATH: ${VTK_DIRECTORY} and ${prefix}")
            endif()
            message(STATUS "Using VTK: ${prefix}")
            # Add the prefix to the list
            list(APPEND VTK_DIRECTORY "${prefix}")
        endif()
    endforeach()

    # if no VTK_DIRECTORY found, error
    if(NOT VTK_DIRECTORY)
        message(FATAL_ERROR "No VTK installation found in LD_LIBRARY_PATH environment variable")
    endif()
endif()