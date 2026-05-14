#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "splinterp.h"
#include <vector>
#include <complex>

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
    {
        py::gil_scoped_release release;
        splinterp::parallel_interp1(splinterp::interp1_F<double>,
                                  data_ptr, nrows,
                                  x_ptr, npoints,
                                  result_ptr, origin_offset);
    }
    
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
    
    {
        py::gil_scoped_release release;
        splinterp::parallel_interp1_complex(
            splinterp::interp1_F_complex<double>,
            data_ptr, nrows,
            x_ptr, npoints,
            result_ptr, origin_offset
        );
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
    {
        py::gil_scoped_release release;
        splinterp::parallel_interp2(splinterp::interp2_F<double>,
                                  data_ptr, nrows, ncols,
                                  x_ptr, y_ptr, npoints,
                                  result_ptr, origin_offset);
    }
    
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
    if (x.ndim() != y.ndim()) {
        throw std::runtime_error("x, y arrays must have the same shape");
    }
    for (size_t i = 0; i < x.ndim(); ++i) {
        if (x.shape()[i] != y.shape()[i]) {
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
    
    {
        py::gil_scoped_release release;
        splinterp::parallel_interp2_complex(
            splinterp::interp2_F_complex<double>,
            data_ptr, nrows, ncols,
            x_ptr, y_ptr, npoints,
            result_ptr, origin_offset
        );
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
    {
        py::gil_scoped_release release;
        splinterp::parallel_interp3(splinterp::interp3_F<double>, 
                                  data_ptr, nrows, ncols, nlayers, 
                                  x_ptr, y_ptr, z_ptr, npoints, 
                                  result_ptr, origin_offset);
    }
    
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
    
    {
        py::gil_scoped_release release;
        splinterp::parallel_interp3_complex(
            splinterp::interp3_F_complex<double>,
            data_ptr, nrows, ncols, nlayers,
            x_ptr, y_ptr, z_ptr, npoints,
            result_ptr, origin_offset
        );
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
