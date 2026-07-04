#include "irongrad/nn/sequential.hpp"

#include <stdexcept>
#include <utility>

namespace irongrad {
namespace nn {

Sequential::Sequential(std::vector<std::unique_ptr<Module>> modules)
    : modules_(std::move(modules)) {
    for (const auto& module : modules_) {
        if (!module) {
            throw std::invalid_argument("Sequential modules must not contain null modules");
        }
    }
}

void Sequential::add(std::unique_ptr<Module> module) {
    if (!module) {
        throw std::invalid_argument("Sequential module must not be null");
    }

    modules_.push_back(std::move(module));
}

Tensor::Ptr Sequential::forward(const Tensor::Ptr& input) {
    if (!input) {
        throw std::invalid_argument("Sequential input must not be null");
    }

    Tensor::Ptr output = input;
    for (const auto& module : modules_) {
        output = module->forward(output);
    }
    return output;
}

std::vector<Tensor::Ptr> Sequential::parameters() const {
    std::vector<Tensor::Ptr> result;

    for (const auto& module : modules_) {
        auto module_parameters = module->parameters();
        result.insert(result.end(), module_parameters.begin(), module_parameters.end());
    }

    return result;
}

} // namespace nn
} // namespace irongrad
