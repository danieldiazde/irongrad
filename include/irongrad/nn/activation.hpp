#pragma once

#include "irongrad/nn/module.hpp"

namespace irongrad {
namespace nn {

class ReLU : public Module {
public:
    Tensor::Ptr forward(const Tensor::Ptr& input) override;
};

} // namespace nn
} // namespace irongrad
