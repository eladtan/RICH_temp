set(ld_lib_path $ENV{LD_LIBRARY_PATH})

# Split LD_LIBRARY_PATH into a CMake list
string(REPLACE ":" ";" ld_lib_path_list "${ld_lib_path}")

if(NOT HDF5_DIRECTORY)
    # Extract likely HDF5 prefixes
    set(HDF5_DIRECTORY "")
    foreach(dir IN LISTS ld_lib_path_list)
        if(dir MATCHES "/hdf5" OR dir MATCHES "/HDF5")
            string(REGEX REPLACE "/lib(64)?$" "" prefix "${dir}")
            # if there is already a prefix - error, multiple HDF5 installations
            if(HDF5_DIRECTORY)
                message(FATAL_ERROR "Multiple HDF5 installations found in LD_LIBRARY_PATH: ${HDF5_DIRECTORY} and ${prefix}")
            endif()
            message(STATUS "Using HDF5: ${prefix}")
            # Add the prefix to the list
            list(APPEND HDF5_DIRECTORY "${prefix}")
        endif()
    endforeach()

    # if no HDF5_DIRECTORY found, error
    if(NOT HDF5_DIRECTORY)
        message(FATAL_ERROR "No HDF5 installation found in LD_LIBRARY_PATH environment variable")
    endif()
endif()

set(HDF5_INCLUDE "${HDF5_DIRECTORY}/include")
set(HDF5_LIB_DIRECTORY "${HDF5_DIRECTORY}/lib")