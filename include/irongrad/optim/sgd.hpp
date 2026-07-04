#pragma once

#include "irongrad/optim/optimizer.hpp"

#include <vector>

namespace irongrad {

class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor::Ptr> parameters,
        double learning_rate = 0.01,
        double momentum = 0.0,
        bool nesterov = false);

    void step() override;

private:
    double learning_rate_;
    double momentum_;
    bool nesterov_;
    std::vector<std::vector<double>> velocities_;
};

} // namespace irongrad
