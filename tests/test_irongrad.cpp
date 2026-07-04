#include "irongrad/irongrad.hpp"

#include <cmath>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using irongrad::SGD;
using irongrad::Shape;
using irongrad::Tensor;
using irongrad::nn::Linear;
using irongrad::nn::ReLU;
using irongrad::nn::Sequential;

Tensor::Ptr tensor(std::initializer_list<double> data) {
    return Tensor::create(std::vector<double>(data));
}

Tensor::Ptr tensor(Shape shape, std::initializer_list<double> data) {
    return Tensor::create(shape, std::vector<double>(data));
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(double actual, double expected, const std::string& message) {
    if (std::fabs(actual - expected) > 1e-9) {
        throw std::runtime_error(message);
    }
}

void require_near(double actual, double expected, double tolerance, const std::string& message) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

void require_vector_near(const std::vector<double>& actual,
                         const std::vector<double>& expected,
                         const std::string& message) {
    require(actual.size() == expected.size(), message + ": size mismatch");

    for (std::size_t i = 0; i < actual.size(); ++i) {
        require_near(actual[i], expected[i], message + " at index " + std::to_string(i));
    }
}

void require_vector_near(const std::vector<double>& actual,
                         const std::vector<double>& expected,
                         double tolerance,
                         const std::string& message) {
    require(actual.size() == expected.size(), message + ": size mismatch");

    for (std::size_t i = 0; i < actual.size(); ++i) {
        require_near(actual[i], expected[i], tolerance, message + " at index " + std::to_string(i));
    }
}

double scalar_value(const Tensor::Ptr& value) {
    require(value->size() == 1, "gradient check function must return a scalar tensor");
    return value->data()[0];
}

std::vector<double> numerical_gradient(
    Shape shape,
    const std::vector<double>& values,
    const std::function<Tensor::Ptr(const Tensor::Ptr&)>& scalar_forward,
    double epsilon = 1e-6
) {
    std::vector<double> gradient(values.size(), 0.0);

    for (std::size_t i = 0; i < values.size(); ++i) {
        auto plus = values;
        plus[i] += epsilon;

        auto minus = values;
        minus[i] -= epsilon;

        const double y_plus = scalar_value(scalar_forward(Tensor::create(shape, plus)));
        const double y_minus = scalar_value(scalar_forward(Tensor::create(shape, minus)));

        gradient[i] = (y_plus - y_minus) / (2.0 * epsilon);
    }

    return gradient;
}

std::vector<double> analytic_gradient(
    Shape shape,
    const std::vector<double>& values,
    const std::function<Tensor::Ptr(const Tensor::Ptr&)>& scalar_forward
) {
    auto input = Tensor::create(shape, values);
    scalar_forward(input)->backward();
    return input->grad();
}

void test_elementwise_autograd() {
    auto a = tensor({1.0, -1.0, 2.0});
    auto b = tensor({2.0, 3.0, 0.5});
    auto c = tensor({0.5, -0.5, 1.0});

    a->mul(b)->add(c)->relu()->sum()->backward();

    require_vector_near(a->grad(), {2.0, 0.0, 0.5}, "chain a gradient");
    require_vector_near(b->grad(), {1.0, 0.0, 2.0}, "chain b gradient");
    require_vector_near(c->grad(), {1.0, 0.0, 1.0}, "chain c gradient");
}

void test_elementwise_gradient_check() {
    const Shape shape = Shape::vector(3);
    const std::vector<double> a_values = {1.0, -1.0, 2.0};
    const std::vector<double> b_values = {2.0, 3.0, 0.5};
    const std::vector<double> c_values = {0.5, -0.5, 1.0};

    auto wrt_a = [&](const Tensor::Ptr& a) {
        auto b = Tensor::create(b_values);
        auto c = Tensor::create(c_values);
        return a->mul(b)->add(c)->relu()->sum();
    };

    auto wrt_b = [&](const Tensor::Ptr& b) {
        auto a = Tensor::create(a_values);
        auto c = Tensor::create(c_values);
        return a->mul(b)->add(c)->relu()->sum();
    };

    auto wrt_c = [&](const Tensor::Ptr& c) {
        auto a = Tensor::create(a_values);
        auto b = Tensor::create(b_values);
        return a->mul(b)->add(c)->relu()->sum();
    };

    require_vector_near(
        analytic_gradient(shape, a_values, wrt_a),
        numerical_gradient(shape, a_values, wrt_a),
        1e-5,
        "elementwise finite-difference gradient for a"
    );
    require_vector_near(
        analytic_gradient(shape, b_values, wrt_b),
        numerical_gradient(shape, b_values, wrt_b),
        1e-5,
        "elementwise finite-difference gradient for b"
    );
    require_vector_near(
        analytic_gradient(shape, c_values, wrt_c),
        numerical_gradient(shape, c_values, wrt_c),
        1e-5,
        "elementwise finite-difference gradient for c"
    );
}

void test_repeated_tensor_use_in_single_operation() {
    auto x = tensor({2.0, -3.0});

    x->mul(x)->sum()->backward();

    require_vector_near(x->grad(), {4.0, -6.0}, "x*x repeated-use gradient");
}

void test_branching_graph_accumulates_paths() {
    auto x = tensor({2.0, -3.0});

    auto square = x->mul(x);
    square->add(x)->sum()->backward();

    require_vector_near(x->grad(), {5.0, -5.0}, "x*x + x branching gradient");
}

void test_branching_graph_gradient_check() {
    const Shape shape = Shape::vector(3);
    const std::vector<double> values = {2.0, -3.0, 0.5};

    auto scalar_forward = [](const Tensor::Ptr& x) {
        return x->mul(x)->add(x)->sum();
    };

    require_vector_near(
        analytic_gradient(shape, values, scalar_forward),
        numerical_gradient(shape, values, scalar_forward),
        1e-5,
        "x*x + x finite-difference gradient"
    );
}

void test_leaf_gradients_accumulate_across_backward_calls() {
    auto x = tensor({2.0});

    x->mul(x)->sum()->backward();
    require_vector_near(x->grad(), {4.0}, "first accumulated gradient");

    x->add(x)->sum()->backward();
    require_vector_near(x->grad(), {6.0}, "gradient accumulation across losses");
}

void test_repeated_backward_on_same_graph_accumulates_once_per_call() {
    auto x = tensor({3.0});
    auto loss = x->mul(x)->sum();

    loss->backward();
    require_vector_near(x->grad(), {6.0}, "first backward on retained graph");

    loss->backward();
    require_vector_near(x->grad(), {12.0}, "second backward on retained graph");
}

void test_matmul_autograd() {
    auto a = tensor(Shape(2, 3), {1.0, 2.0, 3.0,
                                  4.0, 5.0, 6.0});
    auto b = tensor(Shape(3, 2), {7.0, 8.0,
                                  9.0, 10.0,
                                  11.0, 12.0});

    auto c = a->matmul(b);
    require_vector_near(c->data(), {58.0, 64.0, 139.0, 154.0}, "matmul output");

    auto c_naive = a->matmul_naive(b);
    require_vector_near(c_naive->data(), c->data(), "naive matmul output");

    c->sum()->backward();

    require_vector_near(a->grad(), {15.0, 19.0, 23.0,
                                    15.0, 19.0, 23.0}, "matmul lhs gradient");
    require_vector_near(b->grad(), {5.0, 5.0,
                                    7.0, 7.0,
                                    9.0, 9.0}, "matmul rhs gradient");
}

void test_matmul_gradient_check() {
    const Shape a_shape(2, 3);
    const Shape b_shape(3, 2);
    const std::vector<double> a_values = {1.0, 2.0, 3.0,
                                          4.0, 5.0, 6.0};
    const std::vector<double> b_values = {7.0, 8.0,
                                          9.0, 10.0,
                                          11.0, 12.0};

    auto wrt_a = [&](const Tensor::Ptr& a) {
        auto b = Tensor::create(b_shape, b_values);
        return a->matmul(b)->sum();
    };

    auto wrt_b = [&](const Tensor::Ptr& b) {
        auto a = Tensor::create(a_shape, a_values);
        return a->matmul(b)->sum();
    };

    require_vector_near(
        analytic_gradient(a_shape, a_values, wrt_a),
        numerical_gradient(a_shape, a_values, wrt_a),
        1e-5,
        "matmul finite-difference gradient for lhs"
    );
    require_vector_near(
        analytic_gradient(b_shape, b_values, wrt_b),
        numerical_gradient(b_shape, b_values, wrt_b),
        1e-5,
        "matmul finite-difference gradient for rhs"
    );
}

void test_sgd_step() {
    auto weights = tensor({1.0, -2.0});
    weights->set_grad({0.5, -1.0});

    SGD optimizer({weights}, 0.1);
    optimizer.step();

    require_vector_near(weights->data(), {0.95, -1.9}, "sgd update");

    optimizer.zero_grad();
    require_vector_near(weights->grad(), {0.0, 0.0}, "sgd zero grad");
}

void test_linear_layer_autograd() {
    Linear layer(2, 2);
    layer.weights()->set_data({1.0, 2.0,
                               3.0, 4.0});
    layer.bias()->set_data({0.5, -0.5});

    auto input = tensor(Shape(1, 2), {2.0, 3.0});
    auto output = layer.forward(input);

    require_vector_near(output->data(), {11.5, 15.5}, "linear output");

    output->sum()->backward();

    require_vector_near(input->grad(), {3.0, 7.0}, "linear input gradient");
    require_vector_near(layer.weights()->grad(), {2.0, 2.0,
                                                  3.0, 3.0}, "linear weight gradient");
    require_vector_near(layer.bias()->grad(), {1.0, 1.0}, "linear bias gradient");
}

void test_linear_layer_gradient_check() {
    const Shape input_shape(1, 2);

    const std::vector<double> input_values = {2.0, 3.0};
    const std::vector<double> weight_values = {1.0, 2.0,
                                               3.0, 4.0};
    const std::vector<double> bias_values = {0.5, -0.5};

    auto wrt_input = [&](const Tensor::Ptr& input) {
        Linear layer(2, 2);
        layer.weights()->set_data(weight_values);
        layer.bias()->set_data(bias_values);
        return layer.forward(input)->sum();
    };

    auto linear_loss = [&](const std::vector<double>& weights,
                           const std::vector<double>& bias) {
        Linear layer(2, 2);
        layer.weights()->set_data(weights);
        layer.bias()->set_data(bias);
        auto input = Tensor::create(input_shape, input_values);
        return scalar_value(layer.forward(input)->sum());
    };

    auto numerical_parameter_gradient = [&](const std::vector<double>& values,
                                            const std::function<double(const std::vector<double>&)>& loss) {
        std::vector<double> gradient(values.size(), 0.0);
        constexpr double epsilon = 1e-6;

        for (std::size_t i = 0; i < values.size(); ++i) {
            auto plus = values;
            plus[i] += epsilon;

            auto minus = values;
            minus[i] -= epsilon;

            gradient[i] = (loss(plus) - loss(minus)) / (2.0 * epsilon);
        }

        return gradient;
    };

    auto analytic_parameter_gradients = [&]() {
        Linear layer(2, 2);
        layer.weights()->set_data(weight_values);
        layer.bias()->set_data(bias_values);
        auto input = Tensor::create(input_shape, input_values);
        layer.forward(input)->sum()->backward();
        return std::vector<std::vector<double>>{layer.weights()->grad(), layer.bias()->grad()};
    };

    const auto parameter_gradients = analytic_parameter_gradients();

    require_vector_near(
        analytic_gradient(input_shape, input_values, wrt_input),
        numerical_gradient(input_shape, input_values, wrt_input),
        1e-5,
        "linear finite-difference gradient for input"
    );
    require_vector_near(
        parameter_gradients[0],
        numerical_parameter_gradient(weight_values, [&](const std::vector<double>& weights) {
            return linear_loss(weights, bias_values);
        }),
        1e-5,
        "linear finite-difference gradient for weights"
    );
    require_vector_near(
        parameter_gradients[1],
        numerical_parameter_gradient(bias_values, [&](const std::vector<double>& bias) {
            return linear_loss(weight_values, bias);
        }),
        1e-5,
        "linear finite-difference gradient for bias"
    );
}

void test_sequential_model() {
    auto first = std::make_unique<Linear>(2, 2);
    first->weights()->set_data({1.0, -1.0,
                                0.5, 2.0});
    first->bias()->set_data({0.0, 0.0});

    Sequential model;
    model.add(std::move(first));
    model.add(std::make_unique<ReLU>());

    auto input = tensor(Shape(1, 2), {2.0, 3.0});
    auto output = model.forward(input);

    require_vector_near(output->data(), {3.5, 4.0}, "sequential output");
    require(model.parameters().size() == 2, "sequential parameter count");
}

void test_sequential_gradient_check() {
    const Shape input_shape(1, 2);
    const std::vector<double> input_values = {0.75, -0.4};
    const std::vector<double> hidden_weights = {0.6, -0.3,
                                                0.2, 0.9};
    const std::vector<double> hidden_bias = {0.1, -0.2};
    const std::vector<double> output_weights = {0.5,
                                                -0.7};
    const std::vector<double> output_bias = {0.05};

    auto scalar_forward = [&](const Tensor::Ptr& input) {
        auto hidden = std::make_unique<Linear>(2, 2);
        hidden->weights()->set_data(hidden_weights);
        hidden->bias()->set_data(hidden_bias);

        auto output = std::make_unique<Linear>(2, 1);
        output->weights()->set_data(output_weights);
        output->bias()->set_data(output_bias);

        Sequential model;
        model.add(std::move(hidden));
        model.add(std::make_unique<ReLU>());
        model.add(std::move(output));

        return model.forward(input)->sum();
    };

    require_vector_near(
        analytic_gradient(input_shape, input_values, scalar_forward),
        numerical_gradient(input_shape, input_values, scalar_forward),
        1e-5,
        "sequential finite-difference gradient for input"
    );
}

} // namespace

int main() {
    try {
        test_elementwise_autograd();
        test_elementwise_gradient_check();
        test_repeated_tensor_use_in_single_operation();
        test_branching_graph_accumulates_paths();
        test_branching_graph_gradient_check();
        test_leaf_gradients_accumulate_across_backward_calls();
        test_repeated_backward_on_same_graph_accumulates_once_per_call();
        test_matmul_autograd();
        test_matmul_gradient_check();
        test_sgd_step();
        test_linear_layer_autograd();
        test_linear_layer_gradient_check();
        test_sequential_model();
        test_sequential_gradient_check();
    } catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "All IronGrad C++ tests passed.\n";
    return EXIT_SUCCESS;
}
