#include "irongrad/nn/module.hpp"

namespace irongrad {
namespace nn {

std::vector<Tensor::Ptr> Module::parameters() const {
    return {};
}

} // namespace nn
} // namespace irongrad
