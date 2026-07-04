#include "irongrad/irongrad.hpp"

#include <array>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using irongrad::SGD;
using irongrad::Shape;
using irongrad::Tensor;
using irongrad::nn::Linear;

struct Sample {
    double x0;
    double x1;
    double y;
};

class XorModel {
public:
    XorModel() : hidden_(2, 2), output_(2, 1) {
        hidden_.weights()->data() = {
            0.90, 1.10,
            1.05, 0.95,
        };
        hidden_.bias()->data() = {0.05, -0.85};

        output_.weights()->data() = {
            0.80,
            -1.60,
        };
        output_.bias()->data() = {0.10};
    }

    Tensor::Ptr forward(const Tensor::Ptr& input) {
        return output_.forward(hidden_.forward(input)->relu());
    }

    std::vector<Tensor::Ptr> parameters() const {
        auto params = hidden_.parameters();
        auto output_params = output_.parameters();
        params.insert(params.end(), output_params.begin(), output_params.end());
        return params;
    }

private:
    Linear hidden_;
    Linear output_;
};

Tensor::Ptr input_tensor(const Sample& sample) {
    return Tensor::create(Shape(1, 2), {sample.x0, sample.x1});
}

Tensor::Ptr target_tensor(const Sample& sample) {
    return Tensor::create(Shape(1, 1), {-sample.y});
}

Tensor::Ptr squared_error(const Tensor::Ptr& prediction, const Sample& sample) {
    auto error = prediction->add(target_tensor(sample));
    return error->mul(error)->sum();
}

Tensor::Ptr batch_loss(XorModel& model, const std::array<Sample, 4>& samples) {
    Tensor::Ptr total;

    for (const auto& sample : samples) {
        auto prediction = model.forward(input_tensor(sample));
        auto loss = squared_error(prediction, sample);
        total = total ? total->add(loss) : loss;
    }

    return total;
}

} // namespace

int main() {
    const std::array<Sample, 4> samples = {{
        {0.0, 0.0, 0.0},
        {0.0, 1.0, 1.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 0.0},
    }};

    XorModel model;
    SGD optimizer(model.parameters(), 0.03, 0.9);

    constexpr int epochs = 4000;
    constexpr int report_every = 800;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        optimizer.zero_grad();

        auto loss = batch_loss(model, samples);
        const double mean_loss = loss->data()[0] / static_cast<double>(samples.size());

        loss->backward();
        optimizer.step();

        if (epoch == 1 || epoch % report_every == 0) {
            std::cout << "epoch " << std::setw(4) << epoch
                      << " mse=" << std::fixed << std::setprecision(6) << mean_loss << '\n';
        }
    }

    std::cout << "\nXOR predictions:\n";
    for (const auto& sample : samples) {
        auto prediction = model.forward(input_tensor(sample));
        std::cout << static_cast<int>(sample.x0) << " xor "
                  << static_cast<int>(sample.x1) << " -> "
                  << std::fixed << std::setprecision(6) << prediction->data()[0]
                  << " (target " << sample.y << ")\n";
    }

    return 0;
}
