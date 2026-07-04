#include "irongrad/irongrad.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using irongrad::SGD;
using irongrad::Shape;
using irongrad::Tensor;
using irongrad::nn::Linear;

std::vector<double> values(std::size_t count) {
    std::vector<double> result(count);

    for (std::size_t i = 0; i < count; ++i) {
        result[i] = static_cast<double>((i % 17) - 8) / 17.0;
    }

    return result;
}

template <typename Fn>
double time_ms(Fn&& fn, int iterations) {
    const auto start = Clock::now();

    for (int i = 0; i < iterations; ++i) {
        fn();
    }

    const auto end = Clock::now();
    const auto elapsed = std::chrono::duration<double, std::milli>(end - start);
    return elapsed.count() / static_cast<double>(iterations);
}

void print_row(const std::string& name, int iterations, double ms) {
    std::cout << std::left << std::setw(32) << name
              << std::right << std::setw(12) << iterations
              << std::setw(16) << std::fixed << std::setprecision(4) << ms
              << '\n';
}

void benchmark_matmul(std::size_t size, int iterations, bool optimized) {
    auto lhs = Tensor::create(Shape(size, size), values(size * size));
    auto rhs = Tensor::create(Shape(size, size), values(size * size));

    double checksum = 0.0;
    const double ms = time_ms([&]() {
        auto out = optimized ? lhs->matmul(rhs) : lhs->matmul_naive(rhs);
        checksum += out->data()[0];
    }, iterations);

    const std::string label = optimized ? "matmul row-major " : "matmul naive ";
    print_row(label + std::to_string(size) + "x" + std::to_string(size), iterations, ms);

    if (checksum == -1.0) {
        std::cerr << "unreachable checksum guard\n";
    }
}

void benchmark_linear_forward(std::size_t batch,
                              std::size_t input_features,
                              std::size_t output_features,
                              int iterations) {
    Linear layer(input_features, output_features);
    layer.weights()->set_data(values(input_features * output_features));
    layer.bias()->set_data(values(output_features));
    auto input = Tensor::create(Shape(batch, input_features), values(batch * input_features));

    double checksum = 0.0;
    const double ms = time_ms([&]() {
        auto out = layer.forward(input);
        checksum += out->data()[0];
    }, iterations);

    print_row("linear forward " + std::to_string(batch) + "x" + std::to_string(input_features)
                  + " -> " + std::to_string(output_features),
              iterations,
              ms);

    if (checksum == -1.0) {
        std::cerr << "unreachable checksum guard\n";
    }
}

void benchmark_linear_backward(std::size_t batch,
                               std::size_t input_features,
                               std::size_t output_features,
                               int iterations) {
    Linear layer(input_features, output_features);
    layer.weights()->set_data(values(input_features * output_features));
    layer.bias()->set_data(values(output_features));
    auto input = Tensor::create(Shape(batch, input_features), values(batch * input_features));

    double checksum = 0.0;
    const double ms = time_ms([&]() {
        layer.weights()->zero_grad();
        layer.bias()->zero_grad();
        input->zero_grad();

        auto loss = layer.forward(input)->relu()->sum();
        loss->backward();
        checksum += layer.weights()->grad()[0];
    }, iterations);

    print_row("linear fwd+bwd " + std::to_string(batch) + "x" + std::to_string(input_features)
                  + " -> " + std::to_string(output_features),
              iterations,
              ms);

    if (checksum == -1.0) {
        std::cerr << "unreachable checksum guard\n";
    }
}

void benchmark_sgd_step(std::size_t parameters, int iterations) {
    auto weights = Tensor::create(Shape(1, parameters), values(parameters));
    weights->set_grad(values(parameters));
    SGD optimizer({weights}, 0.01, 0.9);

    const double ms = time_ms([&]() {
        optimizer.step();
    }, iterations);

    print_row("sgd step " + std::to_string(parameters) + " params", iterations, ms);
}

} // namespace

int main() {
    std::cout << "IronGrad benchmark\n";
    std::cout << std::left << std::setw(32) << "case"
              << std::right << std::setw(12) << "iterations"
              << std::setw(16) << "ms/iter"
              << '\n';
    std::cout << std::string(60, '-') << '\n';

    benchmark_matmul(32, 100, false);
    benchmark_matmul(32, 100, true);
    benchmark_matmul(64, 30, false);
    benchmark_matmul(64, 30, true);
    benchmark_matmul(128, 8, false);
    benchmark_matmul(128, 8, true);
    benchmark_linear_forward(16, 64, 64, 100);
    benchmark_linear_backward(16, 64, 64, 30);
    benchmark_sgd_step(4096, 500);

    return 0;
}
