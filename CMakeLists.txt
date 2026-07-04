cmake_minimum_required(VERSION 3.20)
project(irongrad LANGUAGES CXX)

add_library(irongrad STATIC
    src/core/tensor.cpp
    src/nn/activation.cpp
    src/nn/linear.cpp
    src/nn/module.cpp
    src/nn/sequential.cpp
    src/optim/optimizer.cpp
    src/optim/sgd.cpp
)

target_compile_features(irongrad PUBLIC cxx_std_20)
target_include_directories(irongrad PUBLIC include)

option(IRONGRAD_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(IRONGRAD_BUILD_TESTS "Build IronGrad tests" ON)
option(IRONGRAD_BUILD_EXAMPLES "Build IronGrad examples" ON)

if(IRONGRAD_ENABLE_SANITIZERS)
    target_compile_options(irongrad PUBLIC -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined)
    target_link_options(irongrad PUBLIC -fsanitize=address,undefined)
endif()

if(IRONGRAD_BUILD_EXAMPLES)
    add_executable(xor_demo examples/xor.cpp)
    target_link_libraries(xor_demo PRIVATE irongrad)
endif()

if(IRONGRAD_BUILD_TESTS)
    enable_testing()

    add_executable(irongrad_tests tests/test_irongrad.cpp)
    target_link_libraries(irongrad_tests PRIVATE irongrad)

    add_test(NAME irongrad_tests COMMAND irongrad_tests)

    if(IRONGRAD_ENABLE_SANITIZERS)
        set_tests_properties(irongrad_tests PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
    endif()
endif()
