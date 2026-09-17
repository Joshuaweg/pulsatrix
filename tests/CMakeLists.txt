add_executable(exai_tests
    device_backend_test.cpp
    cpu_backend_test.cpp
    shape_test.cpp
    assert_test.cpp
)
target_link_libraries(exai_tests PRIVATE exai_core GTest::gtest_main GTest::gmock)

include(GoogleTest)
gtest_discover_tests(exai_tests)
