# IronGrad

IronGrad is a from-scratch C++20 automatic differentiation and neural-network framework.

It implements a small tensor core, reverse-mode autodiff, OOP neural-network modules, SGD, numerical gradient checks, sanitizers, an XOR training demo, and lightweight performance benchmarks.

## Why It Matters

The goal is to build the core pieces of a deep-learning system directly instead of wrapping an existing ML library. The project focuses on:

- Correct reverse-mode autodiff, including branching and repeated-use graphs
- Explicit C++ ownership and mutation APIs
- Numerical verification through finite-difference gradient checks
- Systems-style validation with sanitizers and benchmarks
- A real end-to-end training example

## Architecture

The project is organized as a compiled C++ library with focused domains:

- `include/irongrad/core`: tensor shape, tensor storage, operations, and autograd graph API
- `include/irongrad/optim`: optimizer abstractions and SGD
- `include/irongrad/nn`: module abstraction, layers, activations, and model composition
- `src/`: implementation files for the public headers
- `tests/`: C++ tests that exercise the public API
- `examples/`: runnable training demos
- `benchmarks/`: lightweight performance benchmarks

The top-level public header is:

```cpp
#include "irongrad/irongrad.hpp"
```

## Current Features

- Tensor storage with shape validation
- Explicit tensor mutation APIs: `set_data`, `set_grad`, `mutable_data`, `mutable_grad`
- Elementwise add, multiply, ReLU, sum, row-vector bias addition, and row-major matrix multiplication
- Reverse-mode automatic differentiation
- Naive, row-major, and cache-blocked (tiled) matmul variants with benchmark comparison
- Numerical gradient checks for autograd validation
- Branching/repeated-use graph tests such as `x*x + x`
- Repeated backward tests with leaf-gradient accumulation
- Optimizer base class
- SGD with momentum and optional Nesterov updates
- `nn::Module`, `nn::Linear`, `nn::ReLU`, and `nn::Sequential`
- End-to-end XOR training demo
- AddressSanitizer and UndefinedBehaviorSanitizer test target

## Quick Start

Requires a C++20 compiler.

```bash
make test
make sanitize
make xor
make bench
```

CMake is also supported:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Sanitizers can be enabled through CMake:

```bash
cmake -S . -B build-sanitize -DIRONGRAD_ENABLE_SANITIZERS=ON
cmake --build build-sanitize
ctest --test-dir build-sanitize --output-on-failure
```

## Make Targets

```bash
make build        # builds the C++ test binary
make test         # builds and runs the C++ tests
make sanitize     # runs tests with AddressSanitizer and UndefinedBehaviorSanitizer
make xor          # trains a tiny XOR model end-to-end
make xor-sanitize # runs the XOR demo under sanitizers with leak detection
make bench        # runs lightweight performance benchmarks
make clean        # removes build artifacts
make rebuild      # clean + build
```

## Correctness

The test suite validates autodiff with:

- Closed-form expectations for known operations
- Finite-difference checks that compare autograd gradients against numerical derivatives
- Branching graph cases such as `z = x*x + x`
- Repeated tensor use such as `x*x`
- Repeated backward calls on retained graphs
- Leaf-gradient accumulation across multiple losses
- Graph-lifetime checks that intermediate nodes are released when the graph goes out of scope, guarding against autograd reference cycles

This defends the core claim: gradients are propagated through the graph correctly, including repeated-use paths where small autodiff engines often fail.

## Example Output

`make test`:

```text
All IronGrad C++ tests passed.
```

`make sanitize`:

```text
./build/irongrad_tests_sanitize
All IronGrad C++ tests passed.
```

`make xor`:

```text
epoch    1 mse=0.098000
epoch 4000 mse=0.000000

XOR predictions:
0 xor 0 -> 0.000000 (target 0.000000)
0 xor 1 -> 1.000000 (target 1.000000)
1 xor 0 -> 1.000000 (target 1.000000)
1 xor 1 -> 0.000000 (target 0.000000)
```

`make bench` sample output:

```text
IronGrad benchmark
case                              iterations         ms/iter
------------------------------------------------------------
matmul naive 32x32                       100          0.0176
matmul row-major 32x32                   100          0.0230
matmul tiled 32x32                       100          0.0180
matmul naive 64x64                        30          0.1794
matmul row-major 64x64                    30          0.1505
matmul tiled 64x64                        30          0.1201
matmul naive 128x128                       8          1.7360
matmul row-major 128x128                   8          1.2436
matmul tiled 128x128                       8          1.2563
matmul naive 256x256                       4         16.5784
matmul row-major 256x256                   4          8.9237
matmul tiled 256x256                       4          8.6265
linear forward 16x64 -> 64               100          0.0390
linear fwd+bwd 16x64 -> 64                30          0.1262
sgd step 4096 params                     500          0.0117
```

Benchmark numbers vary by machine, compiler, and current system load. They are intended for local comparison and regression spotting, not as absolute performance claims. The naive variant iterates in the classic `i,j,k` order and pays for scattered RHS access. The row-major variant hoists the LHS scalar and streams contiguous RHS/output rows, which the compiler auto-vectorizes well. The tiled variant applies 32x32 cache blocking; whether it beats row-major at a given size depends on whether the full RHS still fits in L2 (row-major dominates when it does) and on how well the compiler vectorizes the shorter inner loops.

## Release Checklist

- `make test` passes
- `make sanitize` passes
- `make xor` trains the XOR model end-to-end
- `make bench` runs matmul, layer, backward, and optimizer benchmarks
- GitHub CI runs tests, sanitizers, the XOR demo plain and under sanitizers with leak detection, and benchmarks
- The repository is C++20-only
- Build artifacts are ignored and removable with `make clean`
