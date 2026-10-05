# Run by CTest (see tests/CMakeLists.txt): installs this build to a fresh prefix, then
# configures, builds and runs tests/install_consumer against it with find_package alone.
# Inputs: BUILD_DIR, CONSUMER_SOURCE_DIR, WORK_DIR, CONFIG, GENERATOR, CXX_COMPILER.
foreach(var BUILD_DIR CONSUMER_SOURCE_DIR WORK_DIR GENERATOR CXX_COMPILER)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is not set")
    endif()
endforeach()
if(NOT CONFIG)
    set(CONFIG Release)
endif()

function(run)
    execute_process(COMMAND ${ARGV} RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        list(JOIN ARGV " " command)
        message(FATAL_ERROR "Failed (${result}): ${command}")
    endif()
endfunction()

file(REMOVE_RECURSE "${WORK_DIR}")
set(prefix "${WORK_DIR}/prefix")
set(consumer_build "${WORK_DIR}/build")

run("${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${prefix}" --config "${CONFIG}")
run("${CMAKE_COMMAND}" -S "${CONSUMER_SOURCE_DIR}" -B "${consumer_build}" -G "${GENERATOR}"
    "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}" "-DCMAKE_BUILD_TYPE=${CONFIG}" "-DCMAKE_PREFIX_PATH=${prefix}")
run("${CMAKE_COMMAND}" --build "${consumer_build}" --config "${CONFIG}")

find_program(consumer install_consumer PATHS "${consumer_build}" "${consumer_build}/${CONFIG}" NO_DEFAULT_PATH)
if(NOT consumer)
    message(FATAL_ERROR "install_consumer was not built under ${consumer_build}")
endif()
run("${consumer}")
