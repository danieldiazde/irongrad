#include "irongrad/optim/sgd.hpp"

#include <stdexcept>
#include <utility>

namespace irongrad {

SGD::SGD(std::vector<Tensor::Ptr> parameters,
         double learning_rate,
         double momentum,
         bool nesterov)
    : Optimizer(std::move(parameters)),
      learning_rate_(learning_rate),
      momentum_(momentum),
      nesterov_(nesterov) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }

    if (momentum < 0.0) {
        throw std::invalid_argument("Momentum must be non-negative");
    }

    velocities_.reserve(parameters_.size());
    for (const auto& parameter : parameters_) {
        velocities_.push_back(std::vector<double>(parameter->size(), 0.0));
    }
}

void SGD::step() {
    for (std::size_t i = 0; i < parameters_.size(); ++i) {
        auto& parameter = parameters_[i];
        auto& velocity = velocities_[i];

        for (std::size_t j = 0; j < parameter->size(); ++j) {
            double update = parameter->grad()[j];

            if (momentum_ > 0.0) {
                velocity[j] = momentum_ * velocity[j] + update;
                update = nesterov_ ? update + momentum_ * velocity[j] : velocity[j];
            }

            parameter->mutable_data()[j] -= learning_rate_ * update;
        }
    }
}

} // namespace irongrad
