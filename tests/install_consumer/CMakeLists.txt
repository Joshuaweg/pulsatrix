# A separate project that uses an installed pulsatrix the way a downstream user would:
# find_package(pulsatrix) and nothing else. tests/install_consumer_test.cmake installs the
# build tree to a temporary prefix, then configures, builds and runs this project against it.
cmake_minimum_required(VERSION 3.20)
project(pulsatrix_install_consumer CXX)

find_package(pulsatrix 1.0 CONFIG REQUIRED)

# Every installed header must compile on its own include paths, so a header that includes a
# file the install left out, or an SDK the package config didn't find, fails here.
file(GLOB_RECURSE installed_headers RELATIVE "${pulsatrix_INCLUDE_DIR}" "${pulsatrix_INCLUDE_DIR}/pulsatrix/*.hpp")
if(NOT installed_headers)
    message(FATAL_ERROR "No headers found under ${pulsatrix_INCLUDE_DIR}")
endif()
set(all_headers_source "")
foreach(header IN LISTS installed_headers)
    string(APPEND all_headers_source "#include \"${header}\"\n")
endforeach()
file(WRITE "${CMAKE_BINARY_DIR}/all_headers.cpp" "${all_headers_source}")

add_executable(install_consumer main.cpp "${CMAKE_BINARY_DIR}/all_headers.cpp")
target_link_libraries(install_consumer PRIVATE pulsatrix::core)
target_compile_definitions(install_consumer PRIVATE
    EXPECTED_VERSION="${pulsatrix_VERSION}"
    $<$<BOOL:${pulsatrix_HAS_HIP}>:CONSUMER_WITH_HIP>)
