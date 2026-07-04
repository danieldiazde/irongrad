#include "irongrad/nn/activation.hpp"

#include <stdexcept>

namespace irongrad {
namespace nn {

Tensor::Ptr ReLU::forward(const Tensor::Ptr& input) {
    if (!input) {
        throw std::invalid_argument("ReLU input must not be null");
    }

    return input->relu();
}

} // namespace nn
} // namespace irongrad
