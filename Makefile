.PHONY: build test sanitize xor clean rebuild

CXX ?= c++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -Wpedantic -Iinclude
BUILD_DIR := build
TEST_BIN := $(BUILD_DIR)/irongrad_tests
SANITIZE_BIN := $(BUILD_DIR)/irongrad_tests_sanitize
XOR_BIN := $(BUILD_DIR)/xor_demo
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

build:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LIB_SOURCES) $(TEST_SOURCES) -o $(TEST_BIN)

test: build
	./$(TEST_BIN)

sanitize:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SANITIZE_FLAGS) $(LIB_SOURCES) $(TEST_SOURCES) -o $(SANITIZE_BIN)
	ASAN_OPTIONS=detect_leaks=0 ./$(SANITIZE_BIN)

xor:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LIB_SOURCES) $(XOR_SOURCES) -o $(XOR_BIN)
	./$(XOR_BIN)

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build
