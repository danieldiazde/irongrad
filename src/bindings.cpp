#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>
#include "irongrad/tensor.hpp"

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

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor").def(
        py::init([](int rows, int cols, py::array_t<double> input_array) {
            py::buffer_info buf = input_array.request();

            if (buf.size != rows * cols) {
                throw std::invalid_argument("NumPy array size does not match rows*cols");
            }

            double* ptr = static_cast<double*>(buf.ptr);

            std::vector<double> c_data(ptr, ptr + buf.size);

            return Tensor(rows, cols, c_data);
        }))
        .def("matmul", &Tensor::matmul)
        .def_readonly("shape", &Tensor::shape)
        .def_property_readonly("data", [](Tensor& t) -> py::array_t<double> {

            return py::array_t<double>(
                {t.shape[0], t.shape[1]},
                {t.shape[1] * sizeof(double), sizeof(double)},
                t.data.data(),
                py::cast(t) //This way the python gargabge collector doesn't destroy the Tensor as long as the NumpyArray is still used
            );
        })
        .def("backward", &Tensor::backward)
        .def_property_readonly("grad", [](Tensor& t) -> py::array_t<double> {
            return py::array_t<double>(
                {t.shape[0], t.shape[1]},
                {t.shape[1] * sizeof(double), sizeof(double)}, //To move one down it has to paste a row of columns
                t.grad.data(),
                py::cast(t)
            );
        });
        
}