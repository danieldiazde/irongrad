.PHONY: build test clean rebuild

PYTHON      := $(shell uv run python -c "import sys; print(sys.executable)")
PYBIND11_DIR := $(shell uv run python -m pybind11 --cmakedir)

build:
	mkdir -p build
	cd build && cmake .. -DPython_EXECUTABLE=$(PYTHON) -Dpybind11_DIR=$(PYBIND11_DIR) && make

test: build
	uv run pytest tests/ -v

clean:
	rm -rf build

rebuild: clean build
