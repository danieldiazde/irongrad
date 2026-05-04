cmake_minimum_required(VERSION 3.15)
project(irongrad)


set(CMAKE_CXX_STANDARD 20)


find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)


pybind11_add_module(irongrad_backend src/bindings.cpp)