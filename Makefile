.PHONY: build test clean rebuild

PYBIND11_DIR := $(shell uv run python -m pybind11 --cmakedir)

build:
	mkdir -p build
	cd build && cmake .. -Dpybind11_DIR=$(PYBIND11_DIR) && make

test: build
	uv run pytest tests/ -v

clean:
	rm -rf build

rebuild: clean build
