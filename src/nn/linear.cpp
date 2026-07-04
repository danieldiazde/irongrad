#include "irongrad/nn/linear.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace irongrad {
namespace nn {

namespace {

std::size_t require_feature_count(std::size_t value, const char* name) {
    if (value == 0) {
        throw std::invalid_argument(std::string(name) + " feature count must be positive");
    }

    return value;
}

} // namespace

Linear::Linear(std::size_t input_features,
               std::size_t output_features,
               double weight_init,
               double bias_init)
    : input_features_(require_feature_count(input_features, "Input")),
      output_features_(require_feature_count(output_features, "Output")),
      weights_(Tensor::create(Shape(input_features_, output_features_),
                              std::vector<double>(input_features_ * output_features_, weight_init))),
      bias_(Tensor::create(Shape(1, output_features_),
                           std::vector<double>(output_features_, bias_init))) {}

Tensor::Ptr Linear::forward(const Tensor::Ptr& input) {
    if (!input) {
        throw std::invalid_argument("Linear input must not be null");
    }

    if (input->cols() != input_features_) {
        throw std::invalid_argument("Linear input feature count mismatch");
    }

    return input->matmul(weights_)->add_row_vector(bias_);
}

std::vector<Tensor::Ptr> Linear::parameters() const {
    return {weights_, bias_};
}

const Tensor::Ptr& Linear::weights() const {
    return weights_;
}

const Tensor::Ptr& Linear::bias() const {
    return bias_;
}

} // namespace nn
} // namespace irongrad
