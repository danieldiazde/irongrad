#include "irongrad/optim/optimizer.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace irongrad {

Optimizer::Optimizer(std::vector<Tensor::Ptr> parameters)
    : parameters_(std::move(parameters)) {
    for (const auto& parameter : parameters_) {
        if (!parameter) {
            throw std::invalid_argument("Optimizer parameters must not contain null tensors");
        }
    }
}

void Optimizer::zero_grad() {
    for (const auto& parameter : parameters_) {
        parameter->zero_grad();
    }
}

} // namespace irongrad
