#pragma once

#include "irongrad/nn/module.hpp"

#include <cstddef>

namespace irongrad {
namespace nn {

class Linear : public Module {
public:
    Linear(std::size_t input_features,
           std::size_t output_features,
           double weight_init = 0.0,
           double bias_init = 0.0);

    Tensor::Ptr forward(const Tensor::Ptr& input) override;
    std::vector<Tensor::Ptr> parameters() const override;

    const Tensor::Ptr& weights() const;
    const Tensor::Ptr& bias() const;

private:
    std::size_t input_features_;
    std::size_t output_features_;
    Tensor::Ptr weights_;
    Tensor::Ptr bias_;
};

} // namespace nn
} // namespace irongrad
