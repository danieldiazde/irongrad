#include "irongrad/irongrad.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
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

// Every case is calibrated up to a batch of at least this length, so a cheap
// case and an expensive one get the same measurement budget instead of the
// same iteration count.
constexpr double kTargetBatchMs = 200.0;
constexpr int kRepetitions = 7;
constexpr int kMaxIterations = 5000000;
constexpr int kTableWidth = 76;

std::vector<double> values(std::size_t count) {
    std::vector<double> result(count);

    for (std::size_t i = 0; i < count; ++i) {
        result[i] = static_cast<double>((i % 17) - 8) / 17.0;
    }

    return result;
}

struct Measurement {
    int iterations = 0;
    double min_ms = 0.0;
    double median_ms = 0.0;
    double max_ms = 0.0;
    double spread_pct = 0.0;
};

template <typename Fn>
double run_batch_ms(Fn& fn, int iterations) {
    const auto start = Clock::now();

    for (int i = 0; i < iterations; ++i) {
        fn();
    }

    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Ramps the iteration count until a single batch clears the target duration.
// Projects from the observed rate rather than doubling blindly, so the ramp
// costs about one extra batch instead of many.
template <typename Fn>
int calibrate(Fn& fn, double target_ms) {
    int iterations = 1;

    for (;;) {
        const double elapsed = run_batch_ms(fn, iterations);
        if (elapsed >= target_ms || iterations >= kMaxIterations) {
            return iterations;
        }

        const double growth = elapsed > 0.0
            ? std::clamp((target_ms / elapsed) * 1.5, 2.0, 100.0)
            : 100.0;
        const double projected = static_cast<double>(iterations) * growth;

        iterations = projected >= static_cast<double>(kMaxIterations)
            ? kMaxIterations
            : static_cast<int>(projected) + 1;
    }
}

template <typename Fn>
Measurement measure(Fn fn) {
    const int iterations = calibrate(fn, kTargetBatchMs);

    // One untimed batch at the calibrated size, so cold-cache and page-fault
    // costs are paid before the first measured batch.
    run_batch_ms(fn, iterations);

    std::vector<double> samples;
    samples.reserve(kRepetitions);
    for (int repetition = 0; repetition < kRepetitions; ++repetition) {
        samples.push_back(run_batch_ms(fn, iterations) / static_cast<double>(iterations));
    }
    std::sort(samples.begin(), samples.end());

    Measurement result;
    result.iterations = iterations;
    result.min_ms = samples.front();
    result.median_ms = samples[samples.size() / 2];
    result.max_ms = samples.back();
    result.spread_pct = result.median_ms > 0.0
        ? ((result.max_ms - result.min_ms) / result.median_ms) * 100.0
        : 0.0;

    return result;
}

// Two cases are only distinguishable if their observed batch ranges are
// disjoint; overlapping ranges mean the gap is inside the noise.
bool overlaps(const Measurement& lhs, const Measurement& rhs) {
    return lhs.min_ms <= rhs.max_ms && rhs.min_ms <= lhs.max_ms;
}

void print_header() {
    std::cout << std::left << std::setw(32) << "case"
              << std::right << std::setw(10) << "iters"
              << std::setw(12) << "min ms"
              << std::setw(12) << "median ms"
              << std::setw(10) << "spread"
              << '\n';
    std::cout << std::string(kTableWidth, '-') << '\n';
}

void print_row(const std::string& name, const Measurement& measurement) {
    std::cout << std::left << std::setw(32) << name
              << std::right << std::setw(10) << measurement.iterations
              << std::setw(12) << std::fixed << std::setprecision(4) << measurement.min_ms
              << std::setw(12) << std::fixed << std::setprecision(4) << measurement.median_ms
              << std::setw(9) << std::fixed << std::setprecision(1) << measurement.spread_pct
              << "%" << '\n';
}

enum class MatmulKind { Naive, RowMajor, Tiled };

const char* matmul_label(MatmulKind kind) {
    switch (kind) {
    case MatmulKind::Naive:    return "matmul naive ";
    case MatmulKind::RowMajor: return "matmul row-major ";
    case MatmulKind::Tiled:    return "matmul tiled ";
    }
    return "matmul ";
}

Tensor::Ptr matmul_dispatch(const Tensor::Ptr& lhs, const Tensor::Ptr& rhs, MatmulKind kind) {
    switch (kind) {
    case MatmulKind::Naive:    return lhs->matmul_naive(rhs);
    case MatmulKind::RowMajor: return lhs->matmul(rhs);
    case MatmulKind::Tiled:    return lhs->matmul_tiled(rhs);
    }
    return lhs->matmul(rhs);
}

struct MatmulRecord {
    std::size_t size = 0;
    Measurement naive;
    Measurement row_major;
    Measurement tiled;
};

std::string fastest_label(const MatmulRecord& record) {
    struct Candidate {
        const char* name;
        const Measurement* measurement;
    };

    const std::vector<Candidate> candidates = {
        {"naive", &record.naive},
        {"row-major", &record.row_major},
        {"tiled", &record.tiled},
    };

    const Candidate* best = &candidates.front();
    for (const Candidate& candidate : candidates) {
        if (candidate.measurement->min_ms < best->measurement->min_ms) {
            best = &candidate;
        }
    }

    std::vector<std::string> tied;
    for (const Candidate& candidate : candidates) {
        if (overlaps(*candidate.measurement, *best->measurement)) {
            tied.emplace_back(candidate.name);
        }
    }

    if (tied.size() == 1) {
        return tied.front();
    }

    std::string label;
    for (std::size_t i = 0; i < tied.size(); ++i) {
        if (i > 0) {
            label += (i + 1 == tied.size()) ? " and " : ", ";
        }
        label += tied[i];
    }

    return label + " indistinguishable";
}

Measurement benchmark_matmul(std::size_t size, MatmulKind kind) {
    auto lhs = Tensor::create(Shape(size, size), values(size * size));
    auto rhs = Tensor::create(Shape(size, size), values(size * size));

    double checksum = 0.0;
    const Measurement result = measure([&]() {
        auto out = matmul_dispatch(lhs, rhs, kind);
        checksum += out->data()[0];
    });

    print_row(std::string(matmul_label(kind)) + std::to_string(size) + "x" + std::to_string(size),
              result);

    if (checksum == -1.0) {
        std::cerr << "unreachable checksum guard\n";
    }

    return result;
}

void benchmark_linear_forward(std::size_t batch,
                              std::size_t input_features,
                              std::size_t output_features) {
    Linear layer(input_features, output_features);
    layer.weights()->set_data(values(input_features * output_features));
    layer.bias()->set_data(values(output_features));
    auto input = Tensor::create(Shape(batch, input_features), values(batch * input_features));

    double checksum = 0.0;
    const Measurement result = measure([&]() {
        auto out = layer.forward(input);
        checksum += out->data()[0];
    });

    print_row("linear forward " + std::to_string(batch) + "x" + std::to_string(input_features)
                  + " -> " + std::to_string(output_features),
              result);

    if (checksum == -1.0) {
        std::cerr << "unreachable checksum guard\n";
    }
}

void benchmark_linear_backward(std::size_t batch,
                               std::size_t input_features,
                               std::size_t output_features) {
    Linear layer(input_features, output_features);
    layer.weights()->set_data(values(input_features * output_features));
    layer.bias()->set_data(values(output_features));
    auto input = Tensor::create(Shape(batch, input_features), values(batch * input_features));

    double checksum = 0.0;
    const Measurement result = measure([&]() {
        layer.weights()->zero_grad();
        layer.bias()->zero_grad();
        input->zero_grad();

        auto loss = layer.forward(input)->relu()->sum();
        loss->backward();
        checksum += layer.weights()->grad()[0];
    });

    print_row("linear fwd+bwd " + std::to_string(batch) + "x" + std::to_string(input_features)
                  + " -> " + std::to_string(output_features),
              result);

    if (checksum == -1.0) {
        std::cerr << "unreachable checksum guard\n";
    }
}

void benchmark_sgd_step(std::size_t parameters) {
    auto weights = Tensor::create(Shape(1, parameters), values(parameters));
    weights->set_grad(values(parameters));
    SGD optimizer({weights}, 0.01, 0.9);

    const Measurement result = measure([&]() {
        optimizer.step();
    });

    print_row("sgd step " + std::to_string(parameters) + " params", result);
}

// The 512 and 1024 cases are opt-in: naive at 1024 is roughly 64x the work of
// naive at 256, which alone would push a plain `make bench` past its budget.
bool large_matmul_enabled() {
    const char* const setting = std::getenv("IRONGRAD_BENCH_LARGE");
    if (setting == nullptr) {
        return false;
    }

    const std::string value(setting);
    return !value.empty() && value != "0";
}

void print_matmul_summary(const std::vector<MatmulRecord>& records) {
    std::cout << '\n' << "Matmul speedup vs naive (from min ms; higher is faster)" << '\n';
    std::cout << std::left << std::setw(12) << "size"
              << std::right << std::setw(12) << "row-major"
              << std::setw(12) << "tiled"
              << "   " << std::left << "fastest"
              << '\n';
    std::cout << std::string(kTableWidth, '-') << '\n';

    for (const MatmulRecord& record : records) {
        const std::string size_label =
            std::to_string(record.size) + "x" + std::to_string(record.size);

        std::cout << std::left << std::setw(12) << size_label
                  << std::right << std::setw(11) << std::fixed << std::setprecision(2)
                  << (record.naive.min_ms / record.row_major.min_ms) << "x"
                  << std::setw(11) << std::fixed << std::setprecision(2)
                  << (record.naive.min_ms / record.tiled.min_ms) << "x"
                  << "   " << std::left << fastest_label(record)
                  << '\n';
    }
}

} // namespace

int main() {
    const bool large_enabled = large_matmul_enabled();

    std::vector<std::size_t> matmul_sizes = {32, 64, 128, 256};
    if (large_enabled) {
        matmul_sizes.push_back(512);
        matmul_sizes.push_back(1024);
    }

    std::cout << "IronGrad benchmark\n";
    std::cout << "each case: iterations calibrated to >=" << static_cast<int>(kTargetBatchMs)
              << "ms per batch, 1 untimed warmup batch, "
              << kRepetitions << " timed batches\n";
    std::cout << (large_enabled
                      ? "large matmul sizes 512 and 1024 included via IRONGRAD_BENCH_LARGE\n\n"
                      : "large matmul sizes 512 and 1024 skipped; set IRONGRAD_BENCH_LARGE=1 to include them\n\n");
    print_header();

    std::vector<MatmulRecord> matmul_records;
    matmul_records.reserve(matmul_sizes.size());

    for (const std::size_t size : matmul_sizes) {
        MatmulRecord record;
        record.size = size;
        record.naive = benchmark_matmul(size, MatmulKind::Naive);
        record.row_major = benchmark_matmul(size, MatmulKind::RowMajor);
        record.tiled = benchmark_matmul(size, MatmulKind::Tiled);
        matmul_records.push_back(record);
    }

    benchmark_linear_forward(16, 64, 64);
    benchmark_linear_backward(16, 64, 64);
    benchmark_sgd_step(4096);

    print_matmul_summary(matmul_records);

    return 0;
}
