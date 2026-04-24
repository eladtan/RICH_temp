set(ld_lib_path $ENV{LD_LIBRARY_PATH})

# Split LD_LIBRARY_PATH into a CMake list
string(REPLACE ":" ";" ld_lib_path_list "${ld_lib_path}")

if(NOT HDF5_DIRECTORY)
    set(_hdf5_prefixes "")
    foreach(dir IN LISTS ld_lib_path_list)
        if(dir MATCHES "/hdf5" OR dir MATCHES "/HDF5")
            string(REGEX REPLACE "/lib(64)?$" "" prefix "${dir}")
            list(FIND _hdf5_prefixes "${prefix}" _idx)
            if(_idx EQUAL -1)
                list(APPEND _hdf5_prefixes "${prefix}")
                message(STATUS "Found HDF5 candidate: ${prefix}")
            endif()
        endif()
    endforeach()

    list(LENGTH _hdf5_prefixes _hdf5_count)

    if(_hdf5_count EQUAL 0)
        message(FATAL_ERROR "No HDF5 installation found in LD_LIBRARY_PATH")
    elseif(_hdf5_count GREATER 1)
        message(FATAL_ERROR "Multiple distinct HDF5 installations found in LD_LIBRARY_PATH: ${_hdf5_prefixes}")
    endif()

    list(GET _hdf5_prefixes 0 HDF5_DIRECTORY)
    message(STATUS "Using HDF5: ${HDF5_DIRECTORY}")
endif()

set(HDF5_INCLUDE "${HDF5_DIRECTORY}/include")
set(HDF5_LIB_DIRECTORY "${HDF5_DIRECTORY}/lib")