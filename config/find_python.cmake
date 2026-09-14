# Find python3 from $PATH first (module-loaded), then fall back to system.
find_program(PYTHON_EXECUTABLE NAMES python3 NO_DEFAULT_PATH
             PATHS ENV PATH)
if(NOT PYTHON_EXECUTABLE)
    find_program(PYTHON_EXECUTABLE NAMES python3 REQUIRED)
endif()

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    OUTPUT_VARIABLE PYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "from sysconfig import get_paths; print(get_paths()['include'])"
    OUTPUT_VARIABLE PYTHON_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "from sysconfig import get_config_var; print(get_config_var('LIBDIR'))"
    OUTPUT_VARIABLE PYTHON_LIB_DIRECTORY
    OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "from sysconfig import get_config_var; print(get_config_var('LDLIBRARY'))"
    OUTPUT_VARIABLE PYTHON_LIBRARY_FILENAME
    OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REGEX REPLACE "^lib" "" PYTHON_LIB_NAME "${PYTHON_LIBRARY_FILENAME}")
string(REGEX REPLACE "\\.(so|a|dylib).*" "" PYTHON_LIB_NAME "${PYTHON_LIB_NAME}")

# Set PYTHON_LIBRARY so pybind11/FindPythonLibs doesn't override with system python.
find_library(PYTHON_LIBRARY NAMES "${PYTHON_LIBRARY_FILENAME}" "${PYTHON_LIB_NAME}"
             PATHS "${PYTHON_LIB_DIRECTORY}" NO_DEFAULT_PATH)

message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")
message(STATUS "Python version: ${PYTHON_VERSION}")
message(STATUS "Python include dir: ${PYTHON_INCLUDE}")
message(STATUS "Python library dir: ${PYTHON_LIB_DIRECTORY}")
message(STATUS "Python library: ${PYTHON_LIBRARY}")
