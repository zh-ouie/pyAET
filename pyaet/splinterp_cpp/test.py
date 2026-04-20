import numpy as np
from splinterp_cpp import mex_function1, mex_function2, mex_function3
from scipy import io

# 使用1D插值
data_1d = np.array([1.0, 2.0, 3.0, 4.0])
x_1d = np.array([0.5, 1.2, 2.7])
result_1d = mex_function1(data_1d, x_1d)
# print("result_1d\n:", result_1d)
# 使用2D插值
data_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
x_2d = np.array([[0.5, 0.7], [1.2, 1.5]])
y_2d = np.array([[0.3, 0.6], [0.4, 0.8]])
result_2d = mex_function2(data_2d, x_2d, y_2d)
# print("result_2d\n:", result_2d)
# 使用3D插值 (之前已支持)
data_3d = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
x_3d = np.array([[[0.5, 0.7], [1.2, 1.5]], [[0.3, 0.6], [0.4, 0.8]]])
y_3d = np.array([[[0.5, 0.7], [1.2, 1.5]], [[0.3, 0.6], [0.4, 0.8]]])
z_3d = np.array([[[0.5, 0.7], [1.2, 1.5]], [[0.3, 0.6], [0.4, 0.8]]])
result_3d = mex_function3(data_3d, x_3d, y_3d, z_3d)
# print("result_3d\n:", result_3d)


# test raw matlab data
#2D
# % load('test_splinterp2.mat');
# % rot_pj = splinterp2(pj,rot_y,rot_x);
mat = io.loadmat('/Users/longyang/Desktop/try_spinliterp/test_splinterp2.mat')
rot_pj = mat['rot_pj']
rot_x = mat['rot_x']
rot_y = mat['rot_y']
pj = mat['pj']
test_2d = mex_function2(pj, rot_x, rot_y)
test_2d = np.transpose(test_2d, (2, 1, 0))
print(test_2d)

#3D
# load('splinterp3_sample_data.mat');
# pj_cal1 = splinterp3(recK, yj, xj, zj);
mat = io.loadmat('/Users/longyang/Desktop/try_spinliterp/splinterp3_sample_data.mat')
pj_cal1 = mat['pj_cal1']
recK = mat['recK']
yj = mat['yj']
xj = mat['xj']
zj = mat['zj']

test_3d = mex_function3(recK, xj, yj, zj)
# test_3d = mex_function3(recK, xj, zj, yj)
# test_3d = mex_function3(recK, yj, xj, zj)
# test_3d = mex_function3(recK, yj, zj, xj)
# test_3d = mex_function3(recK, zj, xj, yj)
# test_3d = mex_function3(recK, zj, yj, xj)

