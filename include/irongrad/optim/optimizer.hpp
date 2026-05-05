#pragma once
#include "irongrad/tensor.hpp"
#include <vector>
#include <memory>
#include <cmath>

class Optimizer {
protected:
    std::vector<std::shared_ptr<Tensor>> parameters;

public:
    Optimizer(std::vector<std::shared_ptr<Tensor>> params) : parameters(params) {}

    virtual ~Optimizer() = default;

    void zero_grad() {
        for (auto& p : parameters) {
            std::fill(p->grad.begin(), p->grad.end(), 0.0);
        }
    }

    virtual void step() = 0;
};

class SGD : public Optimizer {
private:
    double lr;
    double momentum;
    bool nesterov;

    std::vector<std::vector<double>> velocities;

public:
    SGD(std::vector<std::shared_ptr<Tensor>> params, double lr = 0.01,
        double momentum = 0.0, bool nesterov = false)
        : Optimizer(params), lr(lr), momentum(momentum), nesterov(nesterov) {

        if (momentum > 0.0) {
            for (auto& p : parameters) {
                velocities.push_back(std::vector<double>(p->data.size(), 0.0));
            }
        }
    }

    void step() override {
        for (size_t i = 0; i < parameters.size(); ++i) {
            auto p = parameters[i];
            for (size_t j = 0; j < p->data.size(); ++j) {
                double g = p->grad[j];

                if (momentum > 0.0) {
                    velocities[i][j] = momentum * velocities[i][j] + g;

                    if (nesterov) {
                        g = g + momentum * velocities[i][j];
                    } else {
                        g = velocities[i][j];
                    }
                }

                p->data[j] -= lr * g;
            }
        }
    }
};
