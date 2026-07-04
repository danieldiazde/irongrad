#pragma once

#include "irongrad/core/tensor.hpp"

#include <vector>

namespace irongrad {

class Optimizer {
public:
    explicit Optimizer(std::vector<Tensor::Ptr> parameters);
    virtual ~Optimizer() = default;

    void zero_grad();
    virtual void step() = 0;

protected:
    std::vector<Tensor::Ptr> parameters_;
};

} // namespace irongrad
