#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

std::vector<double> add_arrays(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Size mismatch");
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] + b[i];
    return result;
}

std::vector<double> mul_arrays(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Size mismatch");
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] * b[i];
    return result;
}

double sum_array(const std::vector<double>& a) {
    double acc = 0.0;
    for (double x : a) acc += x;
    return acc;
}

std::vector<double> relu_array(const std::vector<double>& a) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] > 0.0 ? a[i] : 0.0;
    return result;
}

PYBIND11_MODULE(irongrad_backend, m) {
    m.def("add_arrays", &add_arrays);
    m.def("mul_arrays", &mul_arrays);
    m.def("sum_array",  &sum_array);
    m.def("relu_array", &relu_array);
}