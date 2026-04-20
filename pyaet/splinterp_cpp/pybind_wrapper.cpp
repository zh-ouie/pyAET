#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "splinterp.h"

namespace py = pybind11;

// 包装1D插值函数
py::array_t<double> py_mex_function1(py::array_t<double> data, py::array_t<double> x,
                                     long long origin_offset = 1) {
    // 检查输入数组的维度
    if (data.ndim() != 1) {
        throw std::runtime_error("Input data must be 1D");
    }
    
    // 获取数组尺寸
    auto data_shape = data.shape();
    size_t nrows = data_shape[0];
    
    // 获取输出数组的形状和大小
    auto out_shape = x.shape();
    auto out_ndim = x.ndim();
    size_t npoints = 1;
    for (size_t i = 0; i < out_ndim; ++i) {
        npoints *= out_shape[i];
    }
    
    // 创建输出数组
    std::vector<size_t> shape_vec;
    for (size_t i = 0; i < out_ndim; ++i) {
        shape_vec.push_back(out_shape[i]);
    }
    py::array_t<double> result(shape_vec);
    
    // 获取数组的底层数据指针
    py::buffer_info buf_data = data.request();
    py::buffer_info buf_x = x.request();
    py::buffer_info buf_result = result.request();
    
    double* data_ptr = static_cast<double*>(buf_data.ptr);
    double* x_ptr = static_cast<double*>(buf_x.ptr);
    double* result_ptr = static_cast<double*>(buf_result.ptr);
    
    // 调用C++的1D插值函数
    splinterp::parallel_interp1(splinterp::interp1_F<double>,
                              data_ptr, nrows,
                              x_ptr, npoints,
                              result_ptr, origin_offset);
    
    return result;
}

// 包装复数版本的1D插值函数
py::array_t<std::complex<double>> py_mex_function1_complex(
    py::array_t<std::complex<double>> data,
    py::array_t<double> x,
    long long origin_offset = 1) {
    // 检查输入数组的维度
    if (data.ndim() != 1) {
        throw std::runtime_error("Input data must be 1D");
    }
    
    // 获取数组尺寸
    auto data_shape = data.shape();
    size_t nrows = data_shape[0];
    
    // 获取输出数组的形状和大小
    auto out_shape = x.shape();
    auto out_ndim = x.ndim();
    size_t npoints = 1;
    for (size_t i = 0; i < out_ndim; ++i) {
        npoints *= out_shape[i];
    }
    
    // 创建输出数组
    std::vector<size_t> shape_vec;
    for (size_t i = 0; i < out_ndim; ++i) {
        shape_vec.push_back(out_shape[i]);
    }
    py::array_t<std::complex<double>> result(shape_vec);
    
    // 获取数组的底层数据指针
    py::buffer_info buf_data = data.request();
    py::buffer_info buf_x = x.request();
    py::buffer_info buf_result = result.request();
    
    std::complex<double>* data_ptr = static_cast<std::complex<double>*>(buf_data.ptr);
    double* x_ptr = static_cast<double*>(buf_x.ptr);
    std::complex<double>* result_ptr = static_cast<std::complex<double>*>(buf_result.ptr);
    
    // 分离实部和虚部
    std::vector<double> data_r(nrows);
    std::vector<double> data_i(nrows);
    std::vector<double> result_r(npoints);
    std::vector<double> result_i(npoints);
    
    for (size_t i = 0; i < nrows; ++i) {
        data_r[i] = data_ptr[i].real();
        data_i[i] = data_ptr[i].imag();
    }
    
    // 调用C++的复数1D插值函数
    splinterp::parallel_interp1_cx(splinterp::interp1_F_cx<double>,
                                 data_r.data(), data_i.data(),
                                 nrows,
                                 x_ptr, npoints,
                                 result_r.data(), result_i.data(),
                                 origin_offset);
    
    // 合并实部和虚部到复数结果
    for (size_t i = 0; i < npoints; ++i) {
        result_ptr[i] = std::complex<double>(result_r[i], result_i[i]);
    }
    
    return result;
}

// 主要的1D包装函数
py::array py_mex_function1_main(py::array data, py::array_t<double> x,
                               long long origin_offset = 1) {
    // 检查数据类型并调用相应的函数
    if (py::isinstance<py::array_t<std::complex<double>>>(data)) {
        return py_mex_function1_complex(data.cast<py::array_t<std::complex<double>>>(),
                                      x, origin_offset);
    } else {
        return py_mex_function1(data.cast<py::array_t<double>>(),
                              x, origin_offset);
    }
}

// 包装2D插值函数
// 修改2D插值函数中的形状检查逻辑
py::array_t<double> py_mex_function2(py::array_t<double> data, py::array_t<double> x, 
                                     py::array_t<double> y, long long origin_offset = 1) {
    // 检查输入数组的维度
    if (data.ndim() != 2) {
        throw std::runtime_error("Input data must be 2D");
    }
    
    // 获取数组尺寸
    auto data_shape = data.shape();
    size_t nrows = data_shape[0];
    size_t ncols = data_shape[1];
    
    // 检查x, y数组的维度是否相同
    if (x.ndim() != y.ndim()) {
        throw std::runtime_error("x, y arrays must have the same number of dimensions");
    }
    
    // 检查x, y数组的每个维度大小是否相同
    auto x_shape = x.shape();
    auto y_shape = y.shape();
    for (size_t i = 0; i < x.ndim(); ++i) {
        if (x_shape[i] != y_shape[i]) {
            throw std::runtime_error("x, y arrays must have the same shape");
        }
    }
    
    // 获取输出数组的形状和大小
    auto out_shape = x.shape();
    auto out_ndim = x.ndim();
    size_t npoints = 1;
    for (size_t i = 0; i < out_ndim; ++i) {
        npoints *= out_shape[i];
    }
    
    // 创建输出数组
    std::vector<size_t> shape_vec;
    for (size_t i = 0; i < out_ndim; ++i) {
        shape_vec.push_back(out_shape[i]);
    }
    py::array_t<double> result(shape_vec);
    
    // 获取数组的底层数据指针
    py::buffer_info buf_data = data.request();
    py::buffer_info buf_x = x.request();
    py::buffer_info buf_y = y.request();
    py::buffer_info buf_result = result.request();
    
    double* data_ptr = static_cast<double*>(buf_data.ptr);
    double* x_ptr = static_cast<double*>(buf_x.ptr);
    double* y_ptr = static_cast<double*>(buf_y.ptr);
    double* result_ptr = static_cast<double*>(buf_result.ptr);
    
    // 调用C++的2D插值函数
    splinterp::parallel_interp2(splinterp::interp2_F<double>,
                              data_ptr, nrows, ncols,
                              x_ptr, y_ptr, npoints,
                              result_ptr, origin_offset);
    
    return result;
}

// 包装复数版本的2D插值函数
py::array_t<std::complex<double>> py_mex_function2_complex(
    py::array_t<std::complex<double>> data,
    py::array_t<double> x,
    py::array_t<double> y,
    long long origin_offset = 1) {
    // 检查输入数组的维度
    if (data.ndim() != 2) {
        throw std::runtime_error("Input data must be 2D");
    }
    
    // 获取数组尺寸
    auto data_shape = data.shape();
    size_t nrows = data_shape[0];
    size_t ncols = data_shape[1];
    
    // 检查x, y数组的形状是否一致
    if (x.shape() != y.shape()) {
        throw std::runtime_error("x, y arrays must have the same shape");
    }
    
    // 获取输出数组的形状和大小
    auto out_shape = x.shape();
    auto out_ndim = x.ndim();
    size_t npoints = 1;
    for (size_t i = 0; i < out_ndim; ++i) {
        npoints *= out_shape[i];
    }
    
    // 创建输出数组
    std::vector<size_t> shape_vec;
    for (size_t i = 0; i < out_ndim; ++i) {
        shape_vec.push_back(out_shape[i]);
    }
    py::array_t<std::complex<double>> result(shape_vec);
    
    // 获取数组的底层数据指针
    py::buffer_info buf_data = data.request();
    py::buffer_info buf_x = x.request();
    py::buffer_info buf_y = y.request();
    py::buffer_info buf_result = result.request();
    
    std::complex<double>* data_ptr = static_cast<std::complex<double>*>(buf_data.ptr);
    double* x_ptr = static_cast<double*>(buf_x.ptr);
    double* y_ptr = static_cast<double*>(buf_y.ptr);
    std::complex<double>* result_ptr = static_cast<std::complex<double>*>(buf_result.ptr);
    
    // 分离实部和虚部
    std::vector<double> data_r(nrows * ncols);
    std::vector<double> data_i(nrows * ncols);
    std::vector<double> result_r(npoints);
    std::vector<double> result_i(npoints);
    
    for (size_t i = 0; i < nrows * ncols; ++i) {
        data_r[i] = data_ptr[i].real();
        data_i[i] = data_ptr[i].imag();
    }
    
    // 调用C++的复数2D插值函数
    splinterp::parallel_interp2_cx(splinterp::interp2_F_cx<double>,
                                 data_r.data(), data_i.data(),
                                 nrows, ncols,
                                 x_ptr, y_ptr, npoints,
                                 result_r.data(), result_i.data(),
                                 origin_offset);
    
    // 合并实部和虚部到复数结果
    for (size_t i = 0; i < npoints; ++i) {
        result_ptr[i] = std::complex<double>(result_r[i], result_i[i]);
    }
    
    return result;
}

// 主要的2D包装函数
py::array py_mex_function2_main(py::array data, py::array_t<double> x,
                               py::array_t<double> y, long long origin_offset = 1) {
    // 检查数据类型并调用相应的函数
    if (py::isinstance<py::array_t<std::complex<double>>>(data)) {
        return py_mex_function2_complex(data.cast<py::array_t<std::complex<double>>>(),
                                      x, y, origin_offset);
    } else {
        return py_mex_function2(data.cast<py::array_t<double>>(),
                              x, y, origin_offset);
    }
}

// 包装3D插值函数
// 包装3D插值函数
py::array_t<double> py_mex_function3(py::array_t<double> data, py::array_t<double> x, 
                                     py::array_t<double> y, py::array_t<double> z, 
                                     long long origin_offset = 1) {
    // 检查输入数组的维度
    if (data.ndim() != 3) {
        throw std::runtime_error("Input data must be 3D");
    }
    
    // 获取数组尺寸
    auto data_shape = data.shape();
    size_t nrows = data_shape[0];
    size_t ncols = data_shape[1];
    size_t nlayers = data_shape[2];
    
    // 检查x, y, z数组的形状是否一致（修改为手动比较）
    if (x.ndim() != y.ndim() || x.ndim() != z.ndim()) {
        throw std::runtime_error("x, y, z arrays must have the same shape");
    }
    for (size_t i = 0; i < x.ndim(); ++i) {
        if (x.shape()[i] != y.shape()[i] || x.shape()[i] != z.shape()[i]) {
            throw std::runtime_error("x, y, z arrays must have the same shape");
        }
    }
    
    // 获取输出数组的形状和大小
    auto out_shape = x.shape();
    auto out_ndim = x.ndim();
    size_t npoints = 1;
    // 使用循环计算npoints
    for (size_t i = 0; i < out_ndim; ++i) {
        npoints *= out_shape[i];
    }
    
    // 创建输出数组
    std::vector<size_t> shape_vec;
    for (size_t i = 0; i < out_ndim; ++i) {
        shape_vec.push_back(out_shape[i]);
    }
    py::array_t<double> result(shape_vec);
    
    // 获取数组的底层数据指针
    py::buffer_info buf_data = data.request();
    py::buffer_info buf_x = x.request();
    py::buffer_info buf_y = y.request();
    py::buffer_info buf_z = z.request();
    py::buffer_info buf_result = result.request();
    
    double* data_ptr = static_cast<double*>(buf_data.ptr);
    double* x_ptr = static_cast<double*>(buf_x.ptr);
    double* y_ptr = static_cast<double*>(buf_y.ptr);
    double* z_ptr = static_cast<double*>(buf_z.ptr);
    double* result_ptr = static_cast<double*>(buf_result.ptr);
    
    // 调用C++的3D插值函数
    splinterp::parallel_interp3(splinterp::interp3_F<double>, 
                              data_ptr, nrows, ncols, nlayers, 
                              x_ptr, y_ptr, z_ptr, npoints, 
                              result_ptr, origin_offset);
    
    return result;
}

// 包装复数版本的3D插值函数
py::array_t<std::complex<double>> py_mex_function3_complex(
    py::array_t<std::complex<double>> data, 
    py::array_t<double> x, 
    py::array_t<double> y, 
    py::array_t<double> z, 
    long long origin_offset = 1) {
    // 检查输入数组的维度
    if (data.ndim() != 3) {
        throw std::runtime_error("Input data must be 3D");
    }
    
    // 获取数组尺寸
    auto data_shape = data.shape();
    size_t nrows = data_shape[0];
    size_t ncols = data_shape[1];
    size_t nlayers = data_shape[2];
    
    // 检查x, y, z数组的形状是否一致（修改为手动比较）
    if (x.ndim() != y.ndim() || x.ndim() != z.ndim()) {
        throw std::runtime_error("x, y, z arrays must have the same shape");
    }
    for (size_t i = 0; i < x.ndim(); ++i) {
        if (x.shape()[i] != y.shape()[i] || x.shape()[i] != z.shape()[i]) {
            throw std::runtime_error("x, y, z arrays must have the same shape");
        }
    }
    
    // 获取输出数组的形状和大小
    auto out_shape = x.shape();
    auto out_ndim = x.ndim();
    size_t npoints = 1;
    // 使用循环计算npoints
    for (size_t i = 0; i < out_ndim; ++i) {
        npoints *= out_shape[i];
    }
    
    // 创建输出数组
    std::vector<size_t> shape_vec;
    for (size_t i = 0; i < out_ndim; ++i) {
        shape_vec.push_back(out_shape[i]);
    }
    py::array_t<std::complex<double>> result(shape_vec);
    
    // 获取数组的底层数据指针
    py::buffer_info buf_data = data.request();
    py::buffer_info buf_x = x.request();
    py::buffer_info buf_y = y.request();
    py::buffer_info buf_z = z.request();
    py::buffer_info buf_result = result.request();
    
    std::complex<double>* data_ptr = static_cast<std::complex<double>*>(buf_data.ptr);
    double* x_ptr = static_cast<double*>(buf_x.ptr);
    double* y_ptr = static_cast<double*>(buf_y.ptr);
    double* z_ptr = static_cast<double*>(buf_z.ptr);
    std::complex<double>* result_ptr = static_cast<std::complex<double>*>(buf_result.ptr);
    
    // 分离实部和虚部
    std::vector<double> data_r(nrows * ncols * nlayers);
    std::vector<double> data_i(nrows * ncols * nlayers);
    std::vector<double> result_r(npoints);
    std::vector<double> result_i(npoints);
    
    for (size_t i = 0; i < nrows * ncols * nlayers; ++i) {
        data_r[i] = data_ptr[i].real();
        data_i[i] = data_ptr[i].imag();
    }
    
    // 调用C++的复数3D插值函数
    splinterp::parallel_interp3_cx(splinterp::interp3_F_cx<double>,
                                 data_r.data(), data_i.data(),
                                 nrows, ncols, nlayers,
                                 x_ptr, y_ptr, z_ptr, npoints,
                                 result_r.data(), result_i.data(),
                                 origin_offset);
    
    // 合并实部和虚部到复数结果
    for (size_t i = 0; i < npoints; ++i) {
        result_ptr[i] = std::complex<double>(result_r[i], result_i[i]);
    }
    
    return result;
}

// 主要的3D包装函数
py::array py_mex_function3_main(py::array data, py::array_t<double> x, 
                               py::array_t<double> y, py::array_t<double> z, 
                               long long origin_offset = 1) {
    // 检查数据类型并调用相应的函数
    if (py::isinstance<py::array_t<std::complex<double>>>(data)) {
        return py_mex_function3_complex(data.cast<py::array_t<std::complex<double>>>(), 
                                      x, y, z, origin_offset);
    } else {
        return py_mex_function3(data.cast<py::array_t<double>>(), 
                              x, y, z, origin_offset);
    }
}

// 定义Python模块
PYBIND11_MODULE(splinterp_cpp, m) {
    m.doc() = "C++ implementation of splinterp with pybind11";
    m.def("mex_function1", &py_mex_function1_main,
          "Perform 1D interpolation using C++ implementation",
          py::arg("data"), py::arg("x"),
          py::arg("origin_offset") = 1);
    
    // 注册2D插值函数
    m.def("mex_function2", &py_mex_function2_main,
          "Perform 2D interpolation using C++ implementation",
          py::arg("data"), py::arg("x"), py::arg("y"),
          py::arg("origin_offset") = 1);
    
    // 注册3D插值函数
    m.def("mex_function3", &py_mex_function3_main,
          "Perform 3D interpolation using C++ implementation",
          py::arg("data"), py::arg("x"), py::arg("y"), py::arg("z"),
          py::arg("origin_offset") = 1);
}