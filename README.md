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
make bench-large  # adds the 512x512 and 1024x1024 matmul cases
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
each case: iterations calibrated to >=200ms per batch, 1 untimed warmup batch, 7 timed batches
large matmul sizes 512 and 1024 skipped; set IRONGRAD_BENCH_LARGE=1 to include them

case                                 iters      min ms   median ms    spread
----------------------------------------------------------------------------
matmul naive 32x32                   20203      0.0163      0.0165      9.3%
matmul row-major 32x32               10101      0.0197      0.0200      3.4%
matmul tiled 32x32                   10101      0.0199      0.0205      9.1%
matmul naive 64x64                    1771      0.1666      0.1696      3.7%
matmul row-major 64x64                2341      0.1310      0.1345      4.1%
matmul tiled 64x64                    1878      0.1596      0.1622      7.0%
matmul naive 128x128                   203      1.5606      1.5836      2.7%
matmul row-major 128x128               287      0.9801      1.0023     10.1%
matmul tiled 128x128                   225      1.3229      1.3435      6.2%
matmul naive 256x256                    22     14.4437     14.8712      7.9%
matmul row-major 256x256                39      7.8895      7.9822      2.8%
matmul tiled 256x256                    29     10.5125     10.5904      2.7%
linear forward 16x64 -> 64            8170      0.0343      0.0354     10.1%
linear fwd+bwd 16x64 -> 64            2621      0.1098      0.1153     10.6%
sgd step 4096 params                 28453      0.0103      0.0104     10.6%

Matmul speedup vs naive (from min ms; higher is faster)
size           row-major       tiled   fastest
----------------------------------------------------------------------------
32x32              0.83x       0.82x   naive
64x64              1.27x       1.04x   row-major
128x128            1.59x       1.18x   row-major
256x256            1.83x       1.37x   row-major
```

Benchmark numbers vary by machine, compiler, and current system load. They are intended for local comparison and regression spotting, not as absolute performance claims. The naive variant iterates in the classic `i,j,k` order and pays for scattered RHS access. The row-major variant hoists the LHS scalar and streams contiguous RHS/output rows, which the compiler auto-vectorizes well. The tiled variant applies 32x32 cache blocking.

The measured result for tiling is negative. At 32x32 both optimized variants are slower than naive (0.83x and 0.82x): the working set is L1-resident, so there is no locality left to recover and the extra loop bookkeeping is all that is left to pay for. From 64x64 upward row-major is fastest at every size measured, and tiling never overtakes it. `IRONGRAD_BENCH_LARGE=1` extends the sweep to 512x512 and 1024x1024, where the ranking holds: tiled trails row-major by 33-37% at 128 and 256, and by 19-22% at 512 and 1024. The gap does narrow once the RHS stops fitting in L2, which is the effect blocking exists to exploit, but it never closes, and it stops narrowing between 512 and 1024 - both measured 19-22% across two runs. At 1024 row-major reaches 3.76x over naive against tiling's 3.1x. Any crossover where 32x32 blocking starts to pay lies above 1024 on this machine, not inside the 32-256 range measured earlier; row-major's full-width streaming inner loop keeps vectorizing better than the shorter blocked one even at sizes where L2 residency should favor blocking.

## Release Checklist

- `make test` passes
- `make sanitize` passes
- `make xor` trains the XOR model end-to-end
- `make bench` runs matmul, layer, backward, and optimizer benchmarks
- GitHub CI runs tests, sanitizers, the XOR demo plain and under sanitizers with leak detection, and benchmarks
- The repository is C++20-only
- Build artifacts are ignored and removable with `make clean`
