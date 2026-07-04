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

option(IRONGRAD_BUILD_TESTS "Build IronGrad tests" ON)

if(IRONGRAD_BUILD_TESTS)
    enable_testing()

    add_executable(irongrad_tests tests/test_irongrad.cpp)
    target_link_libraries(irongrad_tests PRIVATE irongrad)

    add_test(NAME irongrad_tests COMMAND irongrad_tests)
endif()
