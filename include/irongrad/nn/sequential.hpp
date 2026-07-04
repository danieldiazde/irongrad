#pragma once

#include "irongrad/nn/module.hpp"

#include <memory>
#include <vector>

namespace irongrad {
namespace nn {

class Sequential : public Module {
public:
    Sequential() = default;
    explicit Sequential(std::vector<std::unique_ptr<Module>> modules);

    void add(std::unique_ptr<Module> module);
    Tensor::Ptr forward(const Tensor::Ptr& input) override;
    std::vector<Tensor::Ptr> parameters() const override;

private:
    std::vector<std::unique_ptr<Module>> modules_;
};

} // namespace nn
} // namespace irongrad
