# Search LD_LIBRARY_PATH for a pybind11 installation
set(ld_path "$ENV{LD_LIBRARY_PATH}")
string(REPLACE ":" ";" ld_path_list "${ld_path}")

foreach(dir IN LISTS ld_path_list)
    if(dir MATCHES "[Pp]ybind11")
        string(REGEX REPLACE "/lib(64)?$" "" _prefix "${dir}")
        if(EXISTS "${_prefix}/share/cmake/pybind11")
            set(PYBIND11_DIRECTORY "${_prefix}/share/cmake/pybind11")
            break()
        endif()
    endif()
endforeach()
