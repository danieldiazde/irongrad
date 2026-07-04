.PHONY: build test sanitize clean rebuild

CXX ?= c++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -Wpedantic -Iinclude
BUILD_DIR := build
TEST_BIN := $(BUILD_DIR)/irongrad_tests
SANITIZE_BIN := $(BUILD_DIR)/irongrad_tests_sanitize
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

build:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LIB_SOURCES) $(TEST_SOURCES) -o $(TEST_BIN)

test: build
	./$(TEST_BIN)

sanitize:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SANITIZE_FLAGS) $(LIB_SOURCES) $(TEST_SOURCES) -o $(SANITIZE_BIN)
	ASAN_OPTIONS=detect_leaks=0 ./$(SANITIZE_BIN)

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build
