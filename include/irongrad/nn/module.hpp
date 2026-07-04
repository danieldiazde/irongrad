#pragma once

#include "irongrad/core/tensor.hpp"

#include <vector>

namespace irongrad {
namespace nn {

class Module {
public:
    virtual ~Module() = default;

    virtual Tensor::Ptr forward(const Tensor::Ptr& input) = 0;
    virtual std::vector<Tensor::Ptr> parameters() const;
};

} // namespace nn
} // namespace irongrad
