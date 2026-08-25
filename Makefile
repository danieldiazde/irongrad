.PHONY: build test sanitize xor xor-sanitize bench clean rebuild

CXX ?= c++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -Wpedantic -Iinclude
BENCH_FLAGS ?= -O2 -DNDEBUG
BUILD_DIR := build
TEST_BIN := $(BUILD_DIR)/irongrad_tests
SANITIZE_BIN := $(BUILD_DIR)/irongrad_tests_sanitize
XOR_BIN := $(BUILD_DIR)/xor_demo
XOR_SANITIZE_BIN := $(BUILD_DIR)/xor_demo_sanitize
BENCH_BIN := $(BUILD_DIR)/irongrad_benchmarks
SANITIZE_FLAGS := -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined

LIB_SOURCES := \
	src/core/tensor.cpp \
	src/nn/activation.cpp \
	src/nn/linear.cpp \
	src/nn/module.cpp \
	src/nn/sequential.cpp \
	src/optim/optimizer.cpp \
	src/optim/sgd.cpp

TEST_SOURCES := tests/test_irongrad.cpp
XOR_SOURCES := examples/xor.cpp
BENCH_SOURCES := benchmarks/benchmark_irongrad.cpp

build:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LIB_SOURCES) $(TEST_SOURCES) -o $(TEST_BIN)

test: build
	./$(TEST_BIN)

sanitize:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SANITIZE_FLAGS) $(LIB_SOURCES) $(TEST_SOURCES) -o $(SANITIZE_BIN)
	./$(SANITIZE_BIN)

xor:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LIB_SOURCES) $(XOR_SOURCES) -o $(XOR_BIN)
	./$(XOR_BIN)

xor-sanitize:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SANITIZE_FLAGS) $(LIB_SOURCES) $(XOR_SOURCES) -o $(XOR_SANITIZE_BIN)
	ASAN_OPTIONS=detect_leaks=1 ./$(XOR_SANITIZE_BIN)

bench:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(BENCH_FLAGS) $(LIB_SOURCES) $(BENCH_SOURCES) -o $(BENCH_BIN)
	./$(BENCH_BIN)

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build
