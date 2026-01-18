set(ld_lib_path $ENV{LD_LIBRARY_PATH})

# Split LD_LIBRARY_PATH into a CMake list
string(REPLACE ":" ";" ld_lib_path_list "${ld_lib_path}")

if(NOT PARMETIS_DIRECTORY)
    # Extract likely ParMETIS prefixes
    set(PARMETIS_DIRECTORY "")
    foreach(dir IN LISTS ld_lib_path_list)
        if(dir MATCHES "/ParMETIS")
            string(REGEX REPLACE "/lib(64)?$" "" prefix "${dir}")
        # if there is already a prefix - error, multiple JSON installations
            if(PARMETIS_DIRECTORY)
                message(FATAL_ERROR "Multiple ParMETIS installations found in LD_LIBRARY_PATH: ${PARMETIS_DIRECTORY} and ${prefix}")
            endif()
            message(STATUS "Using ParMETIS: ${prefix}")
            # Add the prefix to the list
            list(APPEND PARMETIS_DIRECTORY "${prefix}")
        endif()
    endforeach()

    # if no PARMETIS_DIRECTORY found, message that it compiles with no ParMETIS
    if(NOT PARMETIS_DIRECTORY)
        message("No ParMETIS installation found in LD_LIBRARY_PATH environment variable")
    else()
        set(PARMETIS_INCLUDE "${PARMETIS_DIRECTORY}/include")
        set(PARMETIS_LIB_DIRECTORY "${PARMETIS_DIRECTORY}/lib")
    endif()
endif()

if(NOT METIS_DIRECTORY)
    # Extract likely METIS prefixes
    set(METIS_DIRECTORY "")
    foreach(dir IN LISTS ld_lib_path_list)
        if(dir MATCHES "/METIS")
            string(REGEX REPLACE "/lib(64)?$" "" prefix "${dir}")
        # if there is already a prefix - error, multiple JSON installations
            if(METIS_DIRECTORY)
                message(FATAL_ERROR "Multiple METIS installations found in LD_LIBRARY_PATH: ${PARMETIS_DIRECTORY} and ${prefix}")
            endif()
            message(STATUS "Using METIS: ${prefix}")
            # Add the prefix to the list
            list(APPEND METIS_DIRECTORY "${prefix}")
        endif()
    endforeach()

    # if no METIS_DIRECTORY found, message that it compiles with no METIS
    if(NOT METIS_DIRECTORY)
        message("No METIS installation found in LD_LIBRARY_PATH environment variable")
    else()
        set(METIS_INCLUDE "${METIS_DIRECTORY}/include")
        set(METIS_LIB_DIRECTORY "${METIS_DIRECTORY}/lib")
    endif()
endif()

if(NOT GKLIB_DIRECTORY)
    # Extract likely GKLIB prefixes
    set(GKLIB_DIRECTORY "")
    foreach(dir IN LISTS ld_lib_path_list)
        if(dir MATCHES "/GKlib")
            string(REGEX REPLACE "/lib(64)?$" "" prefix "${dir}")
        # if there is already a prefix - error, multiple GKLIB installations
            if(GKLIB_DIRECTORY)
                message(FATAL_ERROR "Multiple GKLIB installations found in LD_LIBRARY_PATH: ${GKLIB_DIRECTORY} and ${prefix}")
            endif()
            message(STATUS "Using GKlib: ${prefix}")
            # Add the prefix to the list
            list(APPEND GKLIB_DIRECTORY "${prefix}")
        endif()
    endforeach()

    # if no GKLIB_DIRECTORY found, message that it compiles with no GKLIB
    if(NOT GKLIB_DIRECTORY)
        message("No GKLIB installation found in LD_LIBRARY_PATH environment variable")
    else()
        set(GKLIB_INCLUDE "${GKLIB_DIRECTORY}/include")
        set(GKLIB_LIB_DIRECTORY "${GKLIB_DIRECTORY}/lib64")
    endif()
endif()