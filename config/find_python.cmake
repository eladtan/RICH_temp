# First, find the full path to the python3 executable
find_program(PYTHON_EXECUTABLE NAMES python3 REQUIRED)

# Get Python version (for example, 3.9)
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    OUTPUT_VARIABLE PYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get Python include path
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "from sysconfig import get_paths; print(get_paths()['include'])"
    OUTPUT_VARIABLE PYTHON_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get Python lib directory
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "from sysconfig import get_config_var; print(get_config_var('LIBDIR'))"
    OUTPUT_VARIABLE PYTHON_LIB_DIRECTORY
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get Python lib name (e.g., python3.9)
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "from sysconfig import get_config_var; print(get_config_var('LDLIBRARY'))"
    OUTPUT_VARIABLE PYTHON_LIBRARY_FILENAME
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Optional: extract lib name without 'lib' prefix and .so/.a/.dylib suffix
string(REGEX REPLACE "^lib" "" PYTHON_LIB_NAME "${PYTHON_LIBRARY_FILENAME}")
string(REGEX REPLACE "\\.(so|a|dylib).*" "" PYTHON_LIB_NAME "${PYTHON_LIB_NAME}")

# Output results
message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")
message(STATUS "Python version: ${PYTHON_VERSION}")
message(STATUS "Python include dir: ${PYTHON_INCLUDE}")
message(STATUS "Python library dir: ${PYTHON_LIB_DIRECTORY}")