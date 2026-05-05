# IronGrad

A from-scratch deep learning framework implementing a custom autograd engine, optimizer stack, and transformer model in C++ and Python.

## Motivation

This project was built with the main focus of bypassing PyTorch to deeply comprehend the structure and nature of neural networks, hardware-memory management and LLMs.

## Architecture

Python Frontend + C++ Backend via PyBind11

## Build Instructions

Requires `uv` and `cmake`.

```bash
make build    # compiles the C++ backend into build/
make test     # runs the full pytest suite
make clean    # removes build artifacts
make rebuild  # clean + build
```