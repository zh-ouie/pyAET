import numpy as np
import threading

# Define constants
NUM_THREADS = 6
SERIAL_LIMIT = 4093

def parallel_interp1(f, data, nrows, x, N, result, origin_offset):
    """
    Perform 1D interpolation with threading.

    Args:
        f: Interpolation function.
        data (np.ndarray): 1D array of data.
        nrows (int): Number of rows in the data.
        x (np.ndarray): 1D array of positions to interpolate at.
        N (int): Number of positions to interpolate.
        result (np.ndarray): Array to store the interpolated values.
        origin_offset (int): Offset for origin.
    """
    if N <= SERIAL_LIMIT:
        f(data, nrows, x, N, result, origin_offset)
    else:
        thread_chunk_size = N // NUM_THREADS
        threads = []

        for i in range(NUM_THREADS):
            start_index = i * thread_chunk_size
            end_index = start_index + thread_chunk_size if i < NUM_THREADS - 1 else N
            thread = threading.Thread(target=f, args=(data, nrows, x[start_index:end_index], end_index - start_index, result[start_index:end_index], origin_offset))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()


def parallel_interp1_cx(f, data_r, data_i, nrows, x, N, result_r, result_i, origin_offset):
    """
    Perform 1D complex interpolation with threading.

    Args:
        f: Complex interpolation function.
        data_r (np.ndarray): 1D array of real part of data.
        data_i (np.ndarray): 1D array of imaginary part of data.
        nrows (int): Number of rows in the data.
        x (np.ndarray): 1D array of positions to interpolate at.
        N (int): Number of positions to interpolate.
        result_r (np.ndarray): Array to store the real part of interpolated values.
        result_i (np.ndarray): Array to store the imaginary part of interpolated values.
        origin_offset (int): Offset for origin.
    """
    if N <= SERIAL_LIMIT:
        f(data_r, data_i, nrows, x, N, result_r, result_i, origin_offset)
    else:
        thread_chunk_size = N // NUM_THREADS
        threads = []

        for i in range(NUM_THREADS):
            start_index = i * thread_chunk_size
            end_index = start_index + thread_chunk_size if i < NUM_THREADS - 1 else N
            thread = threading.Thread(target=f, args=(data_r, data_i, nrows, x[start_index:end_index], end_index - start_index, result_r[start_index:end_index], result_i[start_index:end_index], origin_offset))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()





def parallel_interp2(f, data, nrows, ncols, x, y, N, result, origin_offset):
    """
    Perform 2D interpolation with threading.

    Args:
        f: Interpolation function.
        data (np.ndarray): 2D array of data.
        nrows (int): Number of rows in the data.
        ncols (int): Number of columns in the data.
        x (np.ndarray): 1D array of x positions to interpolate at.
        y (np.ndarray): 1D array of y positions to interpolate at.
        N (int): Number of positions to interpolate.
        result (np.ndarray): Array to store the interpolated values.
        origin_offset (int): Offset for origin.
    """
    if N <= SERIAL_LIMIT:
        f(data, nrows, ncols, x, y, N, result, origin_offset)
    else:
        thread_chunk_size = N // NUM_THREADS
        threads = []

        for i in range(NUM_THREADS):
            start_index = i * thread_chunk_size
            end_index = start_index + thread_chunk_size if i < NUM_THREADS - 1 else N
            thread = threading.Thread(target=f, args=(data, nrows, ncols, x[start_index:end_index], y[start_index:end_index], end_index - start_index, result[start_index:end_index], origin_offset))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()




def parallel_interp3(f, data, nrows, ncols, nlayers, x, y, z, N, result, origin_offset):
    """
    Perform 3D interpolation with threading.

    Args:
        f: Interpolation function.
        data (np.ndarray): 3D array of data.
        nrows (int): Number of rows in the data.
        ncols (int): Number of columns in the data.
        nlayers (int): Number of layers in the data.
        x (np.ndarray): 1D array of x positions to interpolate at.
        y (np.ndarray): 1D array of y positions to interpolate at.
        z (np.ndarray): 1D array of z positions to interpolate at.
        N (int): Number of positions to interpolate.
        result (np.ndarray): Array to store the interpolated values.
        origin_offset (int): Offset for origin.
    """
    if N <= SERIAL_LIMIT:
        f(data, nrows, ncols, nlayers, x, y, z, N, result, origin_offset)
    else:
        thread_chunk_size = N // NUM_THREADS
        threads = []

        for i in range(NUM_THREADS):
            start_index = i * thread_chunk_size
            end_index = start_index + thread_chunk_size if i < NUM_THREADS - 1 else N
            thread = threading.Thread(target=f, args=(data, nrows, ncols, nlayers, x[start_index:end_index], y[start_index:end_index], z[start_index:end_index], end_index - start_index, result[start_index:end_index], origin_offset))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()



def parallel_interp3_cx(f, data_r, data_i, nrows, ncols, nlayers, x, y, z, N, result_r, result_i, origin_offset):
    """
    Perform complex 3D interpolation with threading.

    Args:
        f: Interpolation function.
        data_r (np.ndarray): Real part of the 3D array of data.
        data_i (np.ndarray): Imaginary part of the 3D array of data.
        nrows (int): Number of rows in the data.
        ncols (int): Number of columns in the data.
        nlayers (int): Number of layers in the data.
        x (np.ndarray): 1D array of x positions to interpolate at.
        y (np.ndarray): 1D array of y positions to interpolate at.
        z (np.ndarray): 1D array of z positions to interpolate at.
        N (int): Number of positions to interpolate.
        result_r (np.ndarray): Array to store the real part of the interpolated values.
        result_i (np.ndarray): Array to store the imaginary part of the interpolated values.
        origin_offset (int): Offset for origin.
    """
    if N <= SERIAL_LIMIT:
        f(data_r, data_i, nrows, ncols, nlayers, x, y, z, N, result_r, result_i, origin_offset)
    else:
        thread_chunk_size = N // NUM_THREADS
        threads = []

        for i in range(NUM_THREADS):
            start_index = i * thread_chunk_size
            end_index = start_index + thread_chunk_size if i < NUM_THREADS - 1 else N
            thread = threading.Thread(target=f, args=(data_r, data_i, nrows, ncols, nlayers, x[start_index:end_index], y[start_index:end_index], z[start_index:end_index], end_index - start_index, result_r[start_index:end_index], result_i[start_index:end_index], origin_offset))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()



def interp1_F(data, nrows, x, N, origin_offset=0):
    """
    Perform 1D interpolation.

    Args:
        data (numpy.ndarray): 1D array of data values.
        nrows (int): Number of rows in the data array.
        x (numpy.ndarray): 1D array of input values for interpolation.
        N (int): Number of elements in the input array.
        origin_offset (int, optional): Offset for origin. Defaults to 0.

    Returns:
        numpy.ndarray: Interpolated values.
    """
    result = np.zeros(N)
    for i in range(N):
        x_1 = int(np.floor(x[i])) - origin_offset
        if x[i] - origin_offset == nrows - 1:
            x_1 -= 1
        if x_1 < 0 or x_1 + 1 > nrows - 1:
            result[i] = 0
        else:
            f_1 = data[x_1]
            f_2 = data[x_1 + 1]
            w_x1 = x_1 + 1 - (x[i] - origin_offset)
            result[i] = f_1 * w_x1 + f_2 - f_2 * w_x1
    return result

def interp1_F_cx(data_r, data_i, nrows, x, N, origin_offset=0):
    """
    Perform complex 1D interpolation.

    Args:
        data_r (numpy.ndarray): 1D array of real part of complex data.
        data_i (numpy.ndarray): 1D array of imaginary part of complex data.
        nrows (int): Number of rows in the data array.
        x (numpy.ndarray): 1D array of input values for interpolation.
        N (int): Number of elements in the input array.
        origin_offset (int, optional): Offset for origin. Defaults to 0.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Interpolated real and imaginary values.
    """
    result_r = np.zeros(N)
    result_i = np.zeros(N)
    for i in range(N):
        x_1 = int(np.floor(x[i])) - origin_offset
        if x[i] - origin_offset == nrows - 1:
            x_1 -= 1
        if x_1 < 0 or x_1 + 1 > nrows - 1:
            result_r[i] = 0
            result_i[i] = 0
        else:
            f_1_r = data_r[x_1]
            f_2_r = data_r[x_1 + 1]
            w_x1 = x_1 + 1 - (x[i] - origin_offset)
            result_r[i] = f_1_r * w_x1 + f_2_r - f_2_r * w_x1

            f_1_i = data_i[x_1]
            f_2_i = data_i[x_1 + 1]
            result_i[i] = f_1_i * w_x1 + f_2_i - f_2_i * w_x1
    return result_r, result_i


def interp2_F(data, nrows, ncols, x, y, N, origin_offset=0):
    """
    Perform 2D interpolation.

    Args:
        data (numpy.ndarray): 2D array of data values.
        nrows (int): Number of rows in the data array.
        ncols (int): Number of columns in the data array.
        x (numpy.ndarray): 1D array of x-coordinate values for interpolation.
        y (numpy.ndarray): 1D array of y-coordinate values for interpolation.
        N (int): Number of elements in the input arrays.
        origin_offset (int, optional): Offset for origin. Defaults to 0.

    Returns:
        numpy.ndarray: Interpolated values.
    """
    result = np.zeros(N)
    for i in range(N):
        x_1 = int(np.floor(x[i])) - origin_offset
        x_2 = x_1 + 1
        y_1 = int(np.floor(y[i])) - origin_offset
        y_2 = y_1 + 1

        if (x[i] - origin_offset) == (nrows - 1):
            x_2 -= 1
            x_1 -= 1
        if (y[i] - origin_offset) == (ncols - 1):
            y_2 -= 1
            y_1 -= 1

        if x_1 < 0 or x_2 > nrows - 1 or y_1 < 0 or y_2 > ncols - 1:
            result[i] = 0
        else:
            f_11 = data[x_1 + y_1 * nrows]
            f_12 = data[x_1 + y_2 * nrows]
            f_21 = data[x_2 + y_1 * nrows]
            f_22 = data[x_2 + y_2 * nrows]

            w_x1 = x_2 - (x[i] - origin_offset)
            w_x2 = (x[i] - origin_offset) - x_1
            w_y1 = y_2 - (y[i] - origin_offset)
            w_y2 = (y[i] - origin_offset) - y_1

            a = f_11 * w_x1 + f_21 * w_x2
            b = f_12 * w_x1 + f_22 * w_x2
            result[i] = a * w_y1 + b * w_y2

    return result


def interp2_F_cx(data_r, data_i, nrows, ncols, x, y, N, origin_offset=0):
    """
    Perform complex 2D interpolation.

    Args:
        data_r (numpy.ndarray): 2D array of real part of complex data.
        data_i (numpy.ndarray): 2D array of imaginary part of complex data.
        nrows (int): Number of rows in the data array.
        ncols (int): Number of columns in the data array.
        x (numpy.ndarray): 1D array of x-coordinate values for interpolation.
        y (numpy.ndarray): 1D array of y-coordinate values for interpolation.
        N (int): Number of elements in the input arrays.
        origin_offset (int, optional): Offset for origin. Defaults to 0.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Interpolated real and imaginary values.
    """
    result_r = np.zeros(N)
    result_i = np.zeros(N)
    for i in range(N):
        x_1 = int(np.floor(x[i])) - origin_offset
        x_2 = x_1 + 1
        y_1 = int(np.floor(y[i])) - origin_offset
        y_2 = y_1 + 1

        if (x[i] - origin_offset) == (nrows - 1):
            x_2 -= 1
            x_1 -= 1
        if (y[i] - origin_offset) == (ncols - 1):
            y_2 -= 1
            y_1 -= 1

        if x_1 < 0 or x_2 > nrows - 1 or y_1 < 0 or y_2 > ncols - 1:
            result_r[i] = 0
            result_i[i] = 0
        else:
            f_11_r = data_r[x_1 + y_1 * nrows]
            f_12_r = data_r[x_1 + y_2 * nrows]
            f_21_r = data_r[x_2 + y_1 * nrows]
            f_22_r = data_r[x_2 + y_2 * nrows]

            f_11_i = data_i[x_1 + y_1 * nrows]
            f_12_i = data_i[x_1 + y_2 * nrows]
            f_21_i = data_i[x_2 + y_1 * nrows]
            f_22_i = data_i[x_2 + y_2 * nrows]

            w_x1 = x_2 - (x[i] - origin_offset)
            w_x2 = (x[i] - origin_offset) - x_1
            w_y1 = y_2 - (y[i] - origin_offset)
            w_y2 = (y[i] - origin_offset) - y_1

            a = f_11_r * w_x1 + f_21_r * w_x2
            b = f_12_r * w_x1 + f_22_r * w_x2
            result_r[i] = a * w_y1 + b * w_y2

            a = f_11_i * w_x1 + f_21_i * w_x2
            b = f_12_i * w_x1 + f_22_i * w_x2
            result_i[i] = a * w_y1 + b * w_y2

    return result_r, result_i




def interp3_F(data, nrows, ncols, nlayers, x, y, z, N, origin_offset=0):
    """
    Perform 3D interpolation.

    Args:
        data (numpy.ndarray): 3D array of data values.
        nrows (int): Number of rows in the data array.
        ncols (int): Number of columns in the data array.
        nlayers (int): Number of layers in the data array.
        x (numpy.ndarray): 1D array of x-coordinate values for interpolation.
        y (numpy.ndarray): 1D array of y-coordinate values for interpolation.
        z (numpy.ndarray): 1D array of z-coordinate values for interpolation.
        N (int): Number of elements in the input arrays.
        origin_offset (int, optional): Offset for origin. Defaults to 0.

    Returns:
        numpy.ndarray: Interpolated values.
    """
    result = np.zeros(N)
    for i in range(N):
        x_1 = int(np.floor(x[i])) - origin_offset
        x_2 = x_1 + 1
        y_1 = int(np.floor(y[i])) - origin_offset
        y_2 = y_1 + 1
        z_1 = int(np.floor(z[i])) - origin_offset
        z_2 = z_1 + 1

        if (x[i] - origin_offset) == (nrows - 1):
            x_2 -= 1
            x_1 -= 1
        if (y[i] - origin_offset) == (ncols - 1):
            y_2 -= 1
            y_1 -= 1
        if (z[i] - origin_offset) == (nlayers - 1):
            z_2 -= 1
            z_1 -= 1

        if (
            x_1 < 0
            or x_2 > (nrows - 1)
            or y_1 < 0
            or y_2 > (ncols - 1)
            or z_1 < 0
            or z_2 > (nlayers - 1)
        ):
            result[i] = 0
        else:
            layer_size = ncols * nrows
            z_stride = z_1 * layer_size
            y_1_stride = y_1 * nrows
            y_2_stride = y_2 * nrows

            f_11_1 = data[z_stride + y_1_stride + x_1]
            f_12_1 = data[z_stride + y_2_stride + x_1]
            f_21_1 = data[z_stride + y_1_stride + x_2]
            f_22_1 = data[z_stride + y_2_stride + x_2]

            w_x1 = x_2 - (x[i] - origin_offset)
            w_x2 = (x[i] - origin_offset) - x_1
            w_y1 = y_2 - (y[i] - origin_offset)
            w_y2 = (y[i] - origin_offset) - y_1

            a_1 = f_11_1 * w_x1 + f_21_1 * w_x2
            b_1 = f_12_1 * w_x1 + f_22_1 * w_x2

            F_1 = a_1 * w_y1 + b_1 * w_y2

            z_stride = z_2 * layer_size

            f_11_2 = data[z_stride + y_1_stride + x_1]
            f_12_2 = data[z_stride + y_2_stride + x_1]
            f_21_2 = data[z_stride + y_1_stride + x_2]
            f_22_2 = data[z_stride + y_2_stride + x_2]

            a_2 = f_11_2 * w_x1 + f_21_2 * w_x2
            b_2 = f_12_2 * w_x1 + f_22_2 * w_x2

            F_2 = a_2 * w_y1 + b_2 * w_y2

            w_z1 = z_2 - (z[i] - origin_offset)
            w_z2 = (z[i] - origin_offset) - z_1

            result[i] = F_1 * w_z1 + F_2 * w_z2

    return result


def interp3_F_cx(data_r, data_i, nrows, ncols, nlayers, x, y, z, N, origin_offset=0):
    """
    Perform trilinear interpolation of complex data.

    Args:
        data_r (numpy.ndarray): 3D array of real part of complex data.
        data_i (numpy.ndarray): 3D array of imaginary part of complex data.
        nrows (int): Number of rows in the data array.
        ncols (int): Number of columns in the data array.
        nlayers (int): Number of layers in the data array.
        x (numpy.ndarray): 1D array of x-coordinate values for interpolation.
        y (numpy.ndarray): 1D array of y-coordinate values for interpolation.
        z (numpy.ndarray): 1D array of z-coordinate values for interpolation.
        N (int): Number of elements in the input arrays.
        origin_offset (int, optional): Offset for origin. Defaults to 0.

    Returns:
        tuple: Interpolated real part and imaginary part values as separate numpy arrays.
    """
    result_r = np.zeros(N)
    result_i = np.zeros(N)
    for i in range(N):
        x_1 = int(np.floor(x[i])) - origin_offset
        x_2 = x_1 + 1
        y_1 = int(np.floor(y[i])) - origin_offset
        y_2 = y_1 + 1
        z_1 = int(np.floor(z[i])) - origin_offset
        z_2 = z_1 + 1

        if (x[i] - origin_offset) == (nrows - 1):
            x_2 -= 1
            x_1 -= 1
        if (y[i] - origin_offset) == (ncols - 1):
            y_2 -= 1
            y_1 -= 1
        if (z[i] - origin_offset) == (nlayers - 1):
            z_2 -= 1
            z_1 -= 1

        if (
            x_1 < 0
            or x_2 > (nrows - 1)
            or y_1 < 0
            or y_2 > (ncols - 1)
            or z_1 < 0
            or z_2 > (nlayers - 1)
        ):
            result_r[i] = 0
            result_i[i] = 0
        else:
            layer_size = ncols * nrows
            z_stride = z_1 * layer_size
            y_1_stride = y_1 * nrows
            y_2_stride = y_2 * nrows

            w_x1 = x_2 - (x[i] - origin_offset)
            w_x2 = (x[i] - origin_offset) - x_1
            w_y1 = y_2 - (y[i] - origin_offset)
            w_y2 = (y[i] - origin_offset) - y_1

            # Lower Z plane, real part
            f_11_1 = data_r[z_stride + y_1_stride + x_1]
            f_12_1 = data_r[z_stride + y_2_stride + x_1]
            f_21_1 = data_r[z_stride + y_1_stride + x_2]
            f_22_1 = data_r[z_stride + y_2_stride + x_2]

            a_1 = f_11_1 * w_x1 + f_21_1 * w_x2
            b_1 = f_12_1 * w_x1 + f_22_1 * w_x2
            F_1 = a_1 * w_y1 + b_1 * w_y2

            # Upper Z plane, real part
            z_stride = z_2 * layer_size
            f_11_2 = data_r[z_stride + y_1_stride + x_1]
            f_12_2 = data_r[z_stride + y_2_stride + x_1]
            f_21_2 = data_r[z_stride + y_1_stride + x_2]
            f_22_2 = data_r[z_stride + y_2_stride + x_2]

            a_2 = f_11_2 * w_x1 + f_21_2 * w_x2
            b_2 = f_12_2 * w_x1 + f_22_2 * w_x2
            F_2 = a_2 * w_y1 + b_2 * w_y2

            w_z1 = z_2 - (z[i] - origin_offset)
            w_z2 = (z[i] - origin_offset) - z_1

            result_r[i] = F_1 * w_z1 + F_2 * w_z2

            # Lower Z plane, imaginary part
            z_stride = z_1 * layer_size
            f_11_1 = data_i[z_stride + y_1_stride + x_1]
            f_12_1 = data_i[z_stride + y_2_stride + x_1]
            f_21_1 = data_i[z_stride + y_1_stride + x_2]
            f_22_1 = data_i[z_stride + y_2_stride + x_2]

            a_1 = f_11_1 * w_x1 + f_21_1 * w_x2
            b_1 = f_12_1 * w_x1 + f_22_1 * w_x2
            F_1 = a_1 * w_y1 + b_1 * w_y2

            # Upper Z plane, imaginary part
            z_stride = z_2 * layer_size
            f_11_2 = data_i[z_stride + y_1_stride + x_1]
            f_12_2 = data_i[z_stride + y_2_stride + x_1]
            f_21_2 = data_i[z_stride + y_1_stride + x_2]
            f_22_2 = data_i[z_stride + y_2_stride + x_2]

            a_2 = f_11_2 * w_x1 + f_21_2 * w_x2
            b_2 = f_12_2 * w_x1 + f_22_2 * w_x2
            F_2 = a_2 * w_y1 + b_2 * w_y2

            result_i[i] = F_1 * w_z1 + F_2 * w_z2

    return result_r, result_i


# Example usage
# Define your data, x, and result arrays
data = np.array([1, 2, 3, 4, 5])
x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
result = np.empty_like(x)

# Perform interpolation
parallel_interp1(interp1_F, data, len(data), x, len(x), result, 0)
