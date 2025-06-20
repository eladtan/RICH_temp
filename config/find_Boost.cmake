if(NOT DEFINED BOOST_DIR)
    # Try to get BOOST_DIR or BOOST_ROOT from the environment
    if(DEFINED ENV{BOOST_DIR})
        set(BOOST_DIR "$ENV{BOOST_DIR}")
    elseif(DEFINED ENV{BOOST_ROOT})
        set(BOOST_DIR "$ENV{BOOST_ROOT}")
    endif()
endif()

if(NOT BOOST_DIR)
    message(FATAL_ERROR "'-DBOOST_DIR' was not given to CMake, and neither 'BOOST_DIR' nor 'BOOST_ROOT' environment variables are set.")
endif()

message(STATUS "Using Boost: ${BOOST_DIR}")
