# IronGrad

A from-scratch deep learning framework in C++20.

## Motivation

IronGrad is built to explore neural-network internals directly: tensor storage, reverse-mode automatic differentiation, optimizers, model composition, and eventually hardware-aware execution.

## Architecture

The project is organized as a compiled C++ library with small, focused domains:

- `include/irongrad/core`: tensor shape, tensor storage, operations, and autograd graph API
- `include/irongrad/optim`: optimizer abstractions and SGD
- `include/irongrad/nn`: module abstraction, layers, activations, and model composition
- `src/`: implementation files for the public headers
- `tests/`: C++ tests that exercise the public API

The top-level header `include/irongrad/irongrad.hpp` includes the current public framework API.

## Current Features

- Tensor storage with shape validation
- Elementwise add, multiply, ReLU, sum, matrix multiplication, and row-vector bias addition
- Reverse-mode automatic differentiation
- Numerical gradient checks for autograd validation
- Optimizer base class
- SGD with momentum and optional Nesterov updates
- `nn::Module`, `nn::Linear`, `nn::ReLU`, and `nn::Sequential`
- End-to-end XOR training demo

## Correctness

The C++ test suite validates gradients in two ways:

- Closed-form expectations for known operations
- Finite-difference checks that compare autograd gradients against numerical derivatives
- Branching and repeated-use graph cases such as `x*x + x`
- Repeated backward calls on retained graphs, with leaf-gradient accumulation

This catches incorrect backward rules and gives a baseline before adding lower-level performance optimizations.

## Examples

The XOR demo trains a tiny 2-layer ReLU network with IronGrad's tensor, autograd, layer, and SGD APIs:

```bash
make xor
```

## Build Instructions

Requires a C++20 compiler.

```bash
make build    # builds the C++ test binary
make test     # builds and runs the C++ tests
make sanitize # runs tests with AddressSanitizer and UndefinedBehaviorSanitizer
make xor      # trains a tiny XOR model end-to-end
make clean    # removes build artifacts
make rebuild  # clean + build
```

CMake is also supported:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Sanitizers can also be enabled through CMake:

```bash
cmake -S . -B build-sanitize -DIRONGRAD_ENABLE_SANITIZERS=ON
cmake --build build-sanitize
ctest --test-dir build-sanitize --output-on-failure
```

The sanitizer test disables leak detection with `ASAN_OPTIONS=detect_leaks=0` because LeakSanitizer is not reliable in some containerized environments. Address and undefined-behavior checks still run.
