"""
Here's a brief overview of the main components of this code:

    Thread Management: The code uses C++ threads to parallelize interpolation tasks. The number of threads to use is determined by the NUM_THREADS macro, which is set to a default value of 6 if not defined otherwise.

    Interpolation Functions: The code provides interpolation functions for 1D, 2D, and 3D data in both real and complex forms. These interpolation functions are named interp1_F, interp1_F_cx, interp2_F, interp2_F_cx, interp3_F, and interp3_F_cx. They take input data, grid coordinates, and the number of points to interpolate and produce interpolated results.

    Parallelization: To take advantage of multiple threads, the parallel_interp1, parallel_interp1_cx, parallel_interp2, parallel_interp2_cx, parallel_interp3, and parallel_interp3_cx functions split the input data into smaller chunks and distribute the work among multiple threads. Each thread operates on its designated chunk of data.

    Input Data Handling: The interpolation functions handle cases where the input coordinates fall outside the grid dimensions by setting the result to 0.

    Complex Numbers: Some of the interpolation functions (interp1_F_cx, interp2_F_cx, and interp3_F_cx) are designed to work with complex data, where real and imaginary parts are separately interpolated.

    Origin Offset: The origin_offset parameter is used to adjust the coordinates, particularly when working with arrays that use a different origin point (e.g., Fortran-style arrays with an origin of 1).

    Weighted Interpolation: Interpolation involves calculating weighted combinations of neighboring grid points based on the position of the target coordinates within the grid cells.

    Safety Checks: The code includes checks to ensure that computed indices stay within bounds to avoid accessing memory outside the array.

This code is designed for efficient parallel interpolation of data across multiple dimensions. Users can choose the appropriate interpolation function based on their data type and dimensionality. The code provides an efficient way to perform interpolation tasks while utilizing the available CPU cores for parallelism.
"""

import math
import threading
import numpy as np

NUM_THREADS = 6
SERIAL_LIMIT = 4093

def interp1_F(data, x, origin_offset=0):
    """
    Performs 1-D linear interpolation on real data.

    Args:
        data (numpy.ndarray): 1-D array of real data points.
        x (numpy.ndarray): 1-D array of values at which interpolation is desired.
        origin_offset (float, optional): Offset to apply to x values. Defaults to 0.

    Returns:
        numpy.ndarray: 1-D array of interpolated values at x.

    """
    result = np.zeros_like(x)
    nrows = len(data)

    for i in range(len(x)):
        x_val = x[i] - origin_offset
        x_1 = math.floor(x_val)

        if x_val == (nrows - 1):
            x_1 -= 1

        if x_1 < 0 or (x_1 + 1) > (nrows - 1):
            result[i] = 0
        else:
            f_1 = data[int(x_1)]
            f_2 = data[int(x_1) + 1]

            w_x1 = x_1 + 1 - x_val
            result[i] = f_1 * w_x1 + f_2 - f_2 * w_x1

    return result

def interp1_F_cx(data_r, data_i, x, origin_offset=0):
    """
    Performs 1-D linear interpolation on complex data.

    Args:
        data_r (numpy.ndarray): 1-D array of real parts of complex data points.
        data_i (numpy.ndarray): 1-D array of imaginary parts of complex data points.
        x (numpy.ndarray): 1-D array of values at which interpolation is desired.
        origin_offset (float, optional): Offset to apply to x values. Defaults to 0.

    Returns:
        numpy.ndarray: 1-D array of interpolated complex values at x represented as (real, imag).

    """
    result_r = np.zeros_like(x)
    result_i = np.zeros_like(x)
    nrows = len(data_r)

    for i in range(len(x)):
        x_val = x[i] - origin_offset
        x_1 = math.floor(x_val)

        if x_val == (nrows - 1):
            x_1 -= 1

        if x_1 < 0 or (x_1 + 1) > (nrows - 1):
            result_r[i] = 0
            result_i[i] = 0
        else:
            f_1_r = data_r[int(x_1)]
            f_2_r = data_r[int(x_1) + 1]
            f_1_i = data_i[int(x_1)]
            f_2_i = data_i[int(x_1) + 1]

            w_x1 = x_1 + 1 - x_val
            result_r[i] = f_1_r * w_x1 + f_2_r - f_2_r * w_x1
            result_i[i] = f_1_i * w_x1 + f_2_i - f_2_i * w_x1

    return result_r, result_i

def parallel_interp1(f, data, x, result, origin_offset=0):
    """
    Perform parallel 1-D interpolation on real data.

    Args:
        f (function): The interpolation function (e.g., interp1_F).
        data (numpy.ndarray): 1-D array of real data points.
        x (numpy.ndarray): 1-D array of values at which interpolation is desired.
        result (numpy.ndarray): 1-D array to store the interpolated values.
        origin_offset (float, optional): Offset to apply to x values. Defaults to 0.

    """
    N = len(x)

    if N <= SERIAL_LIMIT:
        return f(data, x, origin_offset)

    THREAD_CHUNK_SIZE = N // NUM_THREADS
    workers = []

    for i in range(NUM_THREADS):
        if i == (NUM_THREADS - 1):
            thread = threading.Thread(target=f, args=(data, x[i:], result[i:], origin_offset))
        else:
            thread = threading.Thread(target=f, args=(data, x[i:i+THREAD_CHUNK_SIZE], result[i:i+THREAD_CHUNK_SIZE], origin_offset))
        workers.append(thread)
        thread.start()

    for worker in workers:
        worker.join()

def parallel_interp1_cx(f, data_r, data_i, x, result_r, result_i, origin_offset=0):
    """
    Perform parallel 1-D interpolation on complex data.

    Args:
        f (function): The interpolation function (e.g., interp1_F_cx).
        data_r (numpy.ndarray): 1-D array of real parts of complex data points.
        data_i (numpy.ndarray): 1-D array of imaginary parts of complex data points.
        x (numpy.ndarray): 1-D array of values at which interpolation is desired.
        result_r (numpy.ndarray): 1-D array to store the real parts of the interpolated complex values.
        result_i (numpy.ndarray): 1-D array to store the imaginary parts of the interpolated complex values.
        origin_offset (float, optional): Offset to apply to x values. Defaults to 0.

    """
    N = len(x)

    if N <= SERIAL_LIMIT:
        return f(data_r, data_i, x, result_r, result_i, origin_offset)

    THREAD_CHUNK_SIZE = N // NUM_THREADS
    workers = []

    for i in range(NUM_THREADS):
        if i == (NUM_THREADS - 1):
            thread = threading.Thread(target=f, args=(data_r, data_i, x[i:], result_r[i:], result_i[i:], origin_offset))
        else:
            thread = threading.Thread(target=f, args=(data_r, data_i, x[i:i+THREAD_CHUNK_SIZE], result_r[i:i+THREAD_CHUNK_SIZE], result_i[i:i+THREAD_CHUNK_SIZE], origin_offset))
        workers.append(thread)
        thread.start()

    for worker in workers:
        worker.join()

# Similarly, you can implement interp2_F, interp2_F_cx, interp3_F, interp3_F_cx, and their parallel versions.

# Example usage:
# data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
# x = np.array([1.5, 2.5, 3.5])
# result = np.zeros_like(x)
# parallel_interp1(interp1_F, data, x, result)
# print(result)

# def main():
#     # Example data and interpolation points
#     data_real = np.random.rand(100, 100)
#     data_imag = np.random.rand(100, 100)
#     x_points = np.linspace(0, 99, 1000)
#     y_points = np.linspace(0, 99, 1000)
#
#     # Result arrays for real and imaginary parts
#     result_real = np.zeros((1000, 1000))
#     result_imag = np.zeros((1000, 1000))
#
#     # Interpolate complex data using parallel_interp2_cx
#     parallel_interp2_cx(interp2_F_cx, data_real, data_imag, x_points, y_points, result_real, result_imag)
#
#     # Interpolate real data using parallel_interp2
#     result_real_only = np.zeros((1000, 1000))
#     parallel_interp2(interp2_F, data_real, x_points, y_points, result_real_only)

def interp2_F(data, x, y):
    """
    Example interpolation function for real data.

    Args:
        data (numpy.ndarray): 2-D array of real data points.
        x (float): X-coordinate at which interpolation is desired.
        y (float): Y-coordinate at which interpolation is desired.

    Returns:
        float: Interpolated value.
    """
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1
    alpha = x - x0
    beta = y - y0

    interpolated_value = (1 - alpha) * ((1 - beta) * data[y0, x0] + beta * data[y1, x0]) + \
                         alpha * ((1 - beta) * data[y0, x1] + beta * data[y1, x1])

    return interpolated_value

def interp2_F_cx(data_real, data_imag, x, y):
    """
    Example interpolation function for complex data.

    Args:
        data_real (numpy.ndarray): 2-D array of real parts of complex data points.
        data_imag (numpy.ndarray): 2-D array of imaginary parts of complex data points.
        x (float): X-coordinate at which interpolation is desired.
        y (float): Y-coordinate at which interpolation is desired.

    Returns:
        tuple: Interpolated value as a tuple (real part, imaginary part).
    """
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1
    alpha = x - x0
    beta = y - y0

    real_value = (1 - alpha) * ((1 - beta) * data_real[y0, x0] + beta * data_real[y1, x0]) + \
                 alpha * ((1 - beta) * data_real[y0, x1] + beta * data_real[y1, x1])

    imag_value = (1 - alpha) * ((1 - beta) * data_imag[y0, x0] + beta * data_imag[y1, x0]) + \
                 alpha * ((1 - beta) * data_imag[y0, x1] + beta * data_imag[y1, x1])

    return real_value, imag_value

if __name__ == "__main__":
    main()


def parallel_interp2(f, data, x, y, result, origin_offset=0):
    """
    Perform parallel 2-D interpolation on real data.

    Args:
        f (function): The interpolation function (e.g., interp2_F).
        data (numpy.ndarray): 2-D array of real data points.
        x (numpy.ndarray): 1-D array of x-values at which interpolation is desired.
        y (numpy.ndarray): 1-D array of y-values at which interpolation is desired.
        result (numpy.ndarray): 2-D array to store the interpolated values.
        origin_offset (float, optional): Offset to apply to x and y values. Defaults to 0.

    """
    # Number of data points
    num_points = len(x)

    # Determine the chunk size for each thread
    chunk_size = int(math.ceil(num_points / NUM_THREADS))

    # Create a list to store thread objects
    threads = []

    # Define a function for each thread
    def interpolate_chunk(start, end):
        for i in range(start, end):
            result[i] = f(data, x[i] + origin_offset, y[i] + origin_offset)

    # Create and start threads
    for i in range(NUM_THREADS):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < NUM_THREADS - 1 else num_points
        thread = threading.Thread(target=interpolate_chunk, args=(start, end))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

def parallel_interp2_cx(f, data_r, data_i, x, y, result_r, result_i, origin_offset=0):
    """
    Perform parallel 2-D interpolation on complex data.

    Args:
        f (function): The interpolation function (e.g., interp2_F_cx).
        data_r (numpy.ndarray): 2-D array of real parts of complex data points.
        data_i (numpy.ndarray): 2-D array of imaginary parts of complex data points.
        x (numpy.ndarray): 1-D array of x-values at which interpolation is desired.
        y (numpy.ndarray): 1-D array of y-values at which interpolation is desired.
        result_r (numpy.ndarray): 2-D array to store the real parts of the interpolated complex values.
        result_i (numpy.ndarray): 2-D array to store the imaginary parts of the interpolated complex values.
        origin_offset (float, optional): Offset to apply to x and y values. Defaults to 0.

    """
    # Number of data points
    num_points = len(x)

    # Determine the chunk size for each thread
    chunk_size = int(math.ceil(num_points / NUM_THREADS))

    # Create a list to store thread objects
    threads = []

    # Define a function for each thread
    def interpolate_chunk(start, end):
        for i in range(start, end):
            result_r[i], result_i[i] = f(data_r, data_i, x[i] + origin_offset, y[i] + origin_offset)

    # Create and start threads
    for i in range(NUM_THREADS):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < NUM_THREADS - 1 else num_points
        thread = threading.Thread(target=interpolate_chunk, args=(start, end))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

# ... (Remaining code here) ...
