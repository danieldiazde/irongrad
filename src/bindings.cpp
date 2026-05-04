#include <vector>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<float> add_arrays(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Two arrays cannot be of different size");
    }

    size_t size = a.size();
    std::vector<float> c(size);

    for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }

    return c;
}




PYBIND11_MODULE(irongrad_backend, m) {
    m.doc() = "C++ backend for irongrad";

    m.def("add_arrays", &add_arrays, "Adds two arrays element-wise");
}