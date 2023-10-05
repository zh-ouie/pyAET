import math
import multiprocessing

NUM_PROCESSES = 6
SERIAL_LIMIT = 4093

def parallel_interp1(f, data, nrows, x, N, result, origin_offset):
    """
    Perform 1D linear interpolation in parallel using multiple processes.

    Args:
        f (function): The interpolation function to use.
        data (list or numpy array): 1D array containing data values.
        nrows (int): Number of rows in the data array.
        x (list or numpy array): 1D array of target x-coordinates.
        N (int): Number of target values to interpolate.
        result (list or numpy array): An empty list or array to store the interpolated values.
        origin_offset (int): Offset for the origin.

    Returns:
        None (results are stored in the 'result' array).
    """
    if N <= SERIAL_LIMIT:
        f(data, nrows, x, N, result, origin_offset)
    else:
        THREAD_CHUNK_SIZE = N // NUM_PROCESSES
        processes = []

        tmp_x = x
        tmp_result = result

        for i in range(NUM_PROCESSES):
            if i == (NUM_PROCESSES - 1):
                processes.append(
                    multiprocessing.Process(target=f, args=(data, nrows, tmp_x, N - len(tmp_x), tmp_result, origin_offset))
                )
                break
            else:
                processes.append(
                    multiprocessing.Process(target=f, args=(data, nrows, tmp_x, THREAD_CHUNK_SIZE, tmp_result, origin_offset))
                )

            tmp_x += THREAD_CHUNK_SIZE
            tmp_result += THREAD_CHUNK_SIZE

        for process in processes:
            process.start()

        for process in processes:
            process.join()

def parallel_interp1_cx(f, data_r, data_i, nrows, x, N, result_r, result_i, origin_offset):
    """
    Perform complex 1D linear interpolation in parallel using multiple processes.

    Args:
        f (function): The complex interpolation function to use.
        data_r (list or numpy array): 1D array containing real parts of data values.
        data_i (list or numpy array): 1D array containing imaginary parts of data values.
        nrows (int): Number of rows in the data array.
        x (list or numpy array): 1D array of target x-coordinates.
        N (int): Number of target values to interpolate.
        result_r (list or numpy array): An empty list or array to store the interpolated real parts.
        result_i (list or numpy array): An empty list or array to store the interpolated imaginary parts.
        origin_offset (int): Offset for the origin.

    Returns:
        None (results are stored in the 'result_r' and 'result_i' arrays).
    """
    if N <= SERIAL_LIMIT:
        f(data_r, data_i, nrows, x, N, result_r, result_i, origin_offset)
    else:
        THREAD_CHUNK_SIZE = N // NUM_PROCESSES
        processes = []

        tmp_x = x
        tmp_result_r = result_r
        tmp_result_i = result_i

        for i in range(NUM_PROCESSES):
            if i == (NUM_PROCESSES - 1):
                processes.append(
                    multiprocessing.Process(target=f, args=(data_r, data_i, nrows, tmp_x, N - len(tmp_x), tmp_result_r, tmp_result_i, origin_offset))
                )
                break
            else:
                processes.append(
                    multiprocessing.Process(target=f, args=(data_r, data_i, nrows, tmp_x, THREAD_CHUNK_SIZE, tmp_result_r, tmp_result_i, origin_offset))
                )

            tmp_x += THREAD_CHUNK_SIZE
            tmp_result_r += THREAD_CHUNK_SIZE
            tmp_result_i += THREAD_CHUNK_SIZE

        for process in processes:
            process.start()

        for process in processes:
            process.join()





def parallel_interp2(f, data, nrows, ncols, x, y, N, result, origin_offset):
    """
    Perform 2D linear interpolation in parallel using multiple processes.

    Args:
        f (function): The interpolation function to use.
        data (list or numpy array): 2D array containing data values.
        nrows (int): Number of rows in the data array.
        ncols (int): Number of columns in the data array.
        x (list or numpy array): 1D array of target x-coordinates.
        y (list or numpy array): 1D array of target y-coordinates.
        N (int): Number of target values to interpolate.
        result (list or numpy array): An empty list or array to store the interpolated values.
        origin_offset (int): Offset for the origin.

    Returns:
        None (results are stored in the 'result' array).
    """
    if N <= SERIAL_LIMIT:
        f(data, nrows, ncols, x, y, N, result, origin_offset)
    else:
        THREAD_CHUNK_SIZE = N // NUM_PROCESSES
        processes = []

        tmp_x = x
        tmp_y = y
        tmp_result = result

        for i in range(NUM_PROCESSES):
            if i == (NUM_PROCESSES - 1):
                processes.append(
                    multiprocessing.Process(target=f, args=(data, nrows, ncols, tmp_x, tmp_y, N - len(tmp_x), tmp_result, origin_offset))
                )
                break
            else:
                processes.append(
                    multiprocessing.Process(target=f, args=(data, nrows, ncols, tmp_x, tmp_y, THREAD_CHUNK_SIZE, tmp_result, origin_offset))
                )

            tmp_x += THREAD_CHUNK_SIZE
            tmp_y += THREAD_CHUNK_SIZE
            tmp_result += THREAD_CHUNK_SIZE

        for process in processes:
            process.start()

        for process in processes:
            process.join()

def parallel_interp2_cx(f, data_r, data_i, nrows, ncols, x, y, N, result_r, result_i, origin_offset):
    """
    Perform complex 2D linear interpolation in parallel using multiple processes.

    Args:
        f (function): The complex interpolation function to use.
        data_r (list or numpy array): 2D array containing real parts of data values.
        data_i (list or numpy array): 2D array containing imaginary parts of data values.
        nrows (int): Number of rows in the data array.
        ncols (int): Number of columns in the data array.
        x (list or numpy array): 1D array of target x-coordinates.
        y (list or numpy array): 1D array of target y-coordinates.
        N (int): Number of target values to interpolate.
        result_r (list or numpy array): An empty list or array to store the interpolated real parts.
        result_i (list or numpy array): An empty list or array to store the interpolated imaginary parts.
        origin_offset (int): Offset for the origin.

    Returns:
        None (results are stored in the 'result_r' and 'result_i' arrays).
    """
    if N <= SERIAL_LIMIT:
        f(data_r, data_i, nrows, ncols, x, y, N, result_r, result_i, origin_offset)
    else:
        THREAD_CHUNK_SIZE = N // NUM_PROCESSES
        processes = []

        tmp_x = x
        tmp_y = y
        tmp_result_r = result_r
        tmp_result_i = result_i

        for i in range(NUM_PROCESSES):
            if i == (NUM_PROCESSES - 1):
                processes.append(
                    multiprocessing.Process(target=f, args=(data_r, data_i, nrows, ncols, tmp_x, tmp_y, N - len(tmp_x), tmp_result_r, tmp_result_i, origin_offset))
                )
                break
            else:
                processes.append(
                    multiprocessing.Process(target=f, args=(data_r, data_i, nrows, ncols, tmp_x, tmp_y, THREAD_CHUNK_SIZE, tmp_result_r, tmp_result_i, origin_offset))
                )

            tmp_x += THREAD_CHUNK_SIZE
            tmp_y += THREAD_CHUNK_SIZE
            tmp_result_r += THREAD_CHUNK_SIZE
            tmp_result_i += THREAD_CHUNK_SIZE

        for process in processes:
            process.start()

        for process in processes:
            process.join()





def parallel_interp3(f, data, nrows, ncols, nlayers, x, y, z, N, result, origin_offset):
    """
    Perform 3D linear interpolation in parallel using multiple processes.

    Args:
        f (function): The interpolation function to use.
        data (list or numpy array): 3D array containing data values.
        nrows (int): Number of rows in the data array.
        ncols (int): Number of columns in the data array.
        nlayers (int): Number of layers in the data array.
        x (list or numpy array): 1D array of target x-coordinates.
        y (list or numpy array): 1D array of target y-coordinates.
        z (list or numpy array): 1D array of target z-coordinates.
        N (int): Number of target values to interpolate.
        result (list or numpy array): An empty list or array to store the interpolated values.
        origin_offset (int): Offset for the origin.

    Returns:
        None (results are stored in the 'result' array).
    """
    if N <= SERIAL_LIMIT:
        f(data, nrows, ncols, nlayers, x, y, z, N, result, origin_offset)
    else:
        THREAD_CHUNK_SIZE = N // NUM_PROCESSES
        processes = []

        tmp_x = x
        tmp_y = y
        tmp_z = z
        tmp_result = result

        for i in range(NUM_PROCESSES):
            if i == (NUM_PROCESSES - 1):
                processes.append(
                    multiprocessing.Process(target=f, args=(data, nrows, ncols, nlayers, tmp_x, tmp_y, tmp_z, N - len(tmp_x), tmp_result, origin_offset))
                )
                break
            else:
                processes.append(
                    multiprocessing.Process(target=f, args=(data, nrows, ncols, nlayers, tmp_x, tmp_y, tmp_z, THREAD_CHUNK_SIZE, tmp_result, origin_offset))
                )

            tmp_x += THREAD_CHUNK_SIZE
            tmp_y += THREAD_CHUNK_SIZE
            tmp_z += THREAD_CHUNK_SIZE
            tmp_result += THREAD_CHUNK_SIZE

        for process in processes:
            process.start()

        for process in processes:
            process.join()

def parallel_interp3_cx(f, data_r, data_i, nrows, ncols, nlayers, x, y, z, N, result_r, result_i, origin_offset):
    """
    Perform complex 3D linear interpolation in parallel using multiple processes.

    Args:
        f (function): The complex interpolation function to use.
        data_r (list or numpy array): 3D array containing real parts of data values.
        data_i (list or numpy array): 3D array containing imaginary parts of data values.
        nrows (int): Number of rows in the data array.
        ncols (int): Number of columns in the data array.
        nlayers (int): Number of layers in the data array.
        x (list or numpy array): 1D array of target x-coordinates.
        y (list or numpy array): 1D array of target y-coordinates.
        z (list or numpy array): 1D array of target z-coordinates.
        N (int): Number of target values to interpolate.
        result_r (list or numpy array): An empty list or array to store the interpolated real parts.
        result_i (list or numpy array): An empty list or array to store the interpolated imaginary parts.
        origin_offset (int): Offset for the origin.

    Returns:
        None (results are stored in the 'result_r' and 'result_i' arrays).
    """
    if N <= SERIAL_LIMIT:
        f(data_r, data_i, nrows, ncols, nlayers, x, y, z, N, result_r, result_i, origin_offset)
    else:
        THREAD_CHUNK_SIZE = N // NUM_PROCESSES
        processes = []

        tmp_x = x
        tmp_y = y
        tmp_z = z
        tmp_result_r = result_r
        tmp_result_i = result_i

        for i in range(NUM_PROCESSES):
            if i == (NUM_PROCESSES - 1):
                processes.append(
                    multiprocessing.Process(target=f, args=(data_r, data_i, nrows, ncols, nlayers, tmp_x, tmp_y, tmp_z, N - len(tmp_x), tmp_result_r, tmp_result_i, origin_offset))
                )
                break
            else:
                processes.append(
                    multiprocessing.Process(target=f, args=(data_r, data_i, nrows, ncols, nlayers, tmp_x, tmp_y, tmp_z, THREAD_CHUNK_SIZE, tmp_result_r, tmp_result_i, origin_offset))
                )

            tmp_x += THREAD_CHUNK_SIZE
            tmp_y += THREAD_CHUNK_SIZE
            tmp_z += THREAD_CHUNK_SIZE
            tmp_result_r += THREAD_CHUNK_SIZE
            tmp_result_i += THREAD_CHUNK_SIZE

        for process in processes:
            process.start()

        for process in processes:
            process.join()




import math

def interp1_F(data, nrows, x, N, result, origin_offset=0):
    """
    Perform 1D linear interpolation.

    Args:
        data (list or numpy array): 1D array containing data values.
        nrows (int): Number of rows in the data array.
        x (list or numpy array): 1D array of target x-coordinates.
        N (int): Number of target values to interpolate.
        result (list or numpy array): An empty list or array to store the interpolated values.
        origin_offset (int, optional): Offset for the origin. Defaults to 0.

    Returns:
        None (results are stored in the 'result' array).
    """
    for i in range(N):
        # Get coordinates of bounding grid locations
        x_1 = int(math.floor(x[i])) - origin_offset

        # Handle the special case where x is the last element
        if x[i] - origin_offset == nrows - 1:
            x_1 -= 1

        # Return 0 for target values that are out of bounds
        if x_1 < 0 or x_1 + 1 > nrows - 1:
            result[i] = 0
        else:
            # Get the array values
            f_1 = data[x_1]
            f_2 = data[x_1 + 1]

            # Compute weights
            w_x1 = x_1 + 1 - (x[i] - origin_offset)
            result[i] = f_1 * w_x1 + f_2 - f_2 * w_x1

def interp1_F_cx(data_r, data_i, nrows, x, N, result_r, result_i, origin_offset=0):
    """
    Perform complex 1D linear interpolation.

    Args:
        data_r (list or numpy array): 1D array containing real parts of data values.
        data_i (list or numpy array): 1D array containing imaginary parts of data values.
        nrows (int): Number of rows in the data array.
        x (list or numpy array): 1D array of target x-coordinates.
        N (int): Number of target values to interpolate.
        result_r (list or numpy array): An empty list or array to store the interpolated real parts.
        result_i (list or numpy array): An empty list or array to store the interpolated imaginary parts.
        origin_offset (int, optional): Offset for the origin. Defaults to 0.

    Returns:
        None (results are stored in the 'result_r' and 'result_i' arrays).
    """
    for i in range(N):
        # Get coordinates of bounding grid locations
        x_1 = int(math.floor(x[i])) - origin_offset

        # Handle the special case where x is the last element
        if x[i] - origin_offset == nrows - 1:
            x_1 -= 1

        # Return 0 for target values that are out of bounds
        if x_1 < 0 or x_1 + 1 > nrows - 1:
            result_r[i] = 0
            result_i[i] = 0
        else:
            # Get the array values
            f_1_r = data_r[x_1]
            f_2_r = data_r[x_1 + 1]
            f_1_i = data_i[x_1]
            f_2_i = data_i[x_1 + 1]

            # Compute weights
            w_x1 = x_1 + 1 - (x[i] - origin_offset)
            result_r[i] = f_1_r * w_x1 + f_2_r - f_2_r * w_x1
            result_i[i] = f_1_i * w_x1 + f_2_i - f_2_i * w_x1



import math

def interp2_F(data, nrows, ncols, x, y, N, result, origin_offset=0):
    """
    Perform 2D linear interpolation.

    Args:
        data (list or numpy array): 2D array containing data values.
        nrows (int): Number of rows in the data array.
        ncols (int): Number of columns in the data array.
        x (list or numpy array): 1D array of target x-coordinates.
        y (list or numpy array): 1D array of target y-coordinates.
        N (int): Number of target values to interpolate.
        result (list or numpy array): An empty list or array to store the interpolated values.
        origin_offset (int, optional): Offset for the origin. Defaults to 0.

    Returns:
        None (results are stored in the 'result' array).
    """
    for i in range(N):
        # Get coordinates of bounding grid locations
        x_1 = int(math.floor(x[i])) - origin_offset
        x_2 = x_1 + 1
        y_1 = int(math.floor(y[i])) - origin_offset
        y_2 = y_1 + 1

        # Handle special case where x/y is the last element
        if x[i] - origin_offset == nrows - 1:
            x_2 -= 1
            x_1 -= 1
        if y[i] - origin_offset == ncols - 1:
            y_2 -= 1
            y_1 -= 1

        # Return 0 for target values that are out of bounds
        if (
            x_1 < 0
            or x_2 > nrows - 1
            or y_1 < 0
            or y_2 > ncols - 1
        ):
            result[i] = 0
        else:
            # Get the array values
            f_11 = data[x_1 + y_1 * nrows]
            f_12 = data[x_1 + y_2 * nrows]
            f_21 = data[x_2 + y_1 * nrows]
            f_22 = data[x_2 + y_2 * nrows]

            # Compute weights
            w_x1 = x_2 - (x[i] - origin_offset)
            w_x2 = (x[i] - origin_offset) - x_1
            w_y1 = y_2 - (y[i] - origin_offset)
            w_y2 = (y[i] - origin_offset) - y_1

            a = f_11 * w_x1 + f_21 * w_x2
            b = f_12 * w_x1 + f_22 * w_x2
            result[i] = a * w_y1 + b * w_y2

def interp2_F_cx(data_r, data_i, nrows, ncols, x, y, N, result_r, result_i, origin_offset=0):
    """
    Perform complex 2D linear interpolation.

    Args:
        data_r (list or numpy array): 2D array containing real parts of data values.
        data_i (list or numpy array): 2D array containing imaginary parts of data values.
        nrows (int): Number of rows in the data array.
        ncols (int): Number of columns in the data array.
        x (list or numpy array): 1D array of target x-coordinates.
        y (list or numpy array): 1D array of target y-coordinates.
        N (int): Number of target values to interpolate.
        result_r (list or numpy array): An empty list or array to store the interpolated real parts.
        result_i (list or numpy array): An empty list or array to store the interpolated imaginary parts.
        origin_offset (int, optional): Offset for the origin. Defaults to 0.

    Returns:
        None (results are stored in the 'result_r' and 'result_i' arrays).
    """
    for i in range(N):
        # Get coordinates of bounding grid locations
        x_1 = int(math.floor(x[i])) - origin_offset
        x_2 = x_1 + 1
        y_1 = int(math.floor(y[i])) - origin_offset
        y_2 = y_1 + 1

        # Handle special case where x/y is the last element
        if x[i] - origin_offset == nrows - 1:
            x_2 -= 1
            x_1 -= 1
        if y[i] - origin_offset == ncols - 1:
            y_2 -= 1
            y_1 -= 1

        # Return 0 for target values that are out of bounds
        if (
            x_1 < 0
            or x_2 > nrows - 1
            or y_1 < 0
            or y_2 > ncols - 1
        ):
            result_r[i] = 0
            result_i[i] = 0
        else:
            # Get the array values
            f_11_r = data_r[x_1 + y_1 * nrows]
            f_12_r = data_r[x_1 + y_2 * nrows]
            f_21_r = data_r[x_2 + y_1 * nrows]
            f_22_r = data_r[x_2 + y_2 * nrows]

            f_11_i = data_i[x_1 + y_1 * nrows]
            f_12_i = data_i[x_1 + y_2 * nrows]
            f_21_i = data_i[x_2 + y_1 * nrows]
            f_22_i = data_i[x_2 + y_2 * nrows]

            # Compute weights
            w_x1 = x_2 - (x[i] - origin_offset)
            w_x2 = (x[i] - origin_offset) - x_1
            w_y1 = y_2 - (y[i] - origin_offset)
            w_y2 = (y[i] - origin_offset) - y_1

            a_r = f_11_r * w_x1 + f_21_r * w_x2
            b_r = f_12_r * w_x1 + f_22_r * w_x2
            result_r[i] = a_r * w_y1 + b_r * w_y2

            a_i = f_11_i * w_x1 + f_21_i * w_x2
            b_i = f_12_i * w_x1 + f_22_i * w_x2
            result_i[i] = a_i * w_y1 + b_i * w_y2



def interp3_F(data, nrows, ncols, nlayers, x, y, z, N, result, origin_offset=0):
    """
    Perform 3D linear interpolation.

    Args:
        data (list or numpy array): 3D array containing data values.
        nrows (int): Number of rows in each layer of the data array.
        ncols (int): Number of columns in each layer of the data array.
        nlayers (int): Number of layers in the data array.
        x (list or numpy array): 1D array of target x-coordinates.
        y (list or numpy array): 1D array of target y-coordinates.
        z (list or numpy array): 1D array of target z-coordinates.
        N (int): Number of target values to interpolate.
        result (list or numpy array): An empty list or array to store the interpolated values.
        origin_offset (int, optional): Offset for the origin. Defaults to 0.

    Returns:
        None (results are stored in the 'result' array).
    """
    for i in range(N):
        # Get coordinates of bounding grid locations
        x_1 = int(math.floor(x[i])) - origin_offset
        x_2 = x_1 + 1
        y_1 = int(math.floor(y[i])) - origin_offset
        y_2 = y_1 + 1
        z_1 = int(math.floor(z[i])) - origin_offset
        z_2 = z_1 + 1

        # Handle special case where x, y, or z is the last element
        if x[i] - origin_offset == nrows - 1:
            x_2 -= 1
            x_1 -= 1
        if y[i] - origin_offset == ncols - 1:
            y_2 -= 1
            y_1 -= 1
        if z[i] - origin_offset == nlayers - 1:
            z_2 -= 1
            z_1 -= 1

        # Return 0 for target values that are out of bounds
        if (
            x_1 < 0
            or x_2 > nrows - 1
            or y_1 < 0
            or y_2 > ncols - 1
            or z_1 < 0
            or z_2 > nlayers - 1
        ):
            result[i] = 0
        else:
            # Precompute some stride-related constants that are reused
            layer_size = ncols * nrows
            z_1_stride = z_1 * layer_size
            y_1_stride = y_1 * nrows
            y_2_stride = y_2 * nrows

            # Get the array values for the lower z slice, real part
            f_111 = data[z_1_stride + y_1_stride + x_1]
            f_121 = data[z_1_stride + y_2_stride + x_1]
            f_211 = data[z_1_stride + y_1_stride + x_2]
            f_221 = data[z_1_stride + y_2_stride + x_2]

            # Compute weights
            w_x1 = x_2 - (x[i] - origin_offset)
            w_x2 = (x[i] - origin_offset) - x_1
            w_y1 = y_2 - (y[i] - origin_offset)
            w_y2 = (y[i] - origin_offset) - y_1

            a_1, b_1 = f_111 * w_x1 + f_211 * w_x2, f_121 * w_x1 + f_221 * w_x2
            F_1 = a_1 * w_y1 + b_1 * w_y2

            # Update some stride-related constants that are reused
            z_2_stride = z_2 * layer_size

            # Get the array values for the upper z slice, real part
            f_112 = data[z_2_stride + y_1_stride + x_1]
            f_122 = data[z_2_stride + y_2_stride + x_1]
            f_212 = data[z_2_stride + y_1_stride + x_2]
            f_222 = data[z_2_stride + y_2_stride + x_2]

            # Compute weights
            w_z1 = z_2 - (z[i] - origin_offset)
            w_z2 = (z[i] - origin_offset) - z_1

            a_2, b_2 = f_112 * w_x1 + f_212 * w_x2, f_122 * w_x1 + f_222 * w_x2
            F_2 = a_2 * w_y1 + b_2 * w_y2

            result[i] = F_1 * w_z1 + F_2 * w_z2


import math


def interp3_F_cx(data_r, data_i, nrows, ncols, nlayers, x, y, z, N, result_r, result_i, origin_offset=0):
    """
    Perform complex 3D linear interpolation.

    Args:
        data_r (list or numpy array): 3D array containing real parts of data values.
        data_i (list or numpy array): 3D array containing imaginary parts of data values.
        nrows (int): Number of rows in each layer of the data array.
        ncols (int): Number of columns in each layer of the data array.
        nlayers (int): Number of layers in the data array.
        x (list or numpy array): 1D array of target x-coordinates.
        y (list or numpy array): 1D array of target y-coordinates.
        z (list or numpy array): 1D array of target z-coordinates.
        N (int): Number of target values to interpolate.
        result_r (list or numpy array): An empty list or array to store the interpolated real parts.
        result_i (list or numpy array): An empty list or array to store the interpolated imaginary parts.
        origin_offset (int, optional): Offset for the origin. Defaults to 0.

    Returns:
        None (results are stored in the 'result_r' and 'result_i' arrays).
    """
    for i in range(N):
        # Get coordinates of bounding grid locations
        x_1 = int(math.floor(x[i])) - origin_offset
        x_2 = x_1 + 1
        y_1 = int(math.floor(y[i])) - origin_offset
        y_2 = y_1 + 1
        z_1 = int(math.floor(z[i])) - origin_offset
        z_2 = z_1 + 1

        # Handle special case where x, y, or z is the last element
        if x[i] - origin_offset == nrows - 1:
            x_2 -= 1
            x_1 -= 1
        if y[i] - origin_offset == ncols - 1:
            y_2 -= 1
            y_1 -= 1
        if z[i] - origin_offset == nlayers - 1:
            z_2 -= 1
            z_1 -= 1

        # Return 0 for target values that are out of bounds
        if (
                x_1 < 0
                or x_2 > nrows - 1
                or y_1 < 0
                or y_2 > ncols - 1
                or z_1 < 0
                or z_2 > nlayers - 1
        ):
            result_r[i] = 0
            result_i[i] = 0
        else:
            # Precompute some stride-related constants that are reused
            layer_size = ncols * nrows
            z_1_stride = z_1 * layer_size
            y_1_stride = y_1 * nrows
            y_2_stride = y_2 * nrows

            # Compute weights
            w_x1 = x_2 - (x[i] - origin_offset)
            w_x2 = (x[i] - origin_offset) - x_1
            w_y1 = y_2 - (y[i] - origin_offset)
            w_y2 = (y[i] - origin_offset) - y_1

            # Lower Z plane, real part
            f_11_1_r = data_r[z_1_stride + y_1_stride + x_1]
            f_12_1_r = data_r[z_1_stride + y_2_stride + x_1]
            f_21_1_r = data_r[z_1_stride + y_1_stride + x_2]
            f_22_1_r = data_r[z_1_stride + y_2_stride + x_2]

            a_1_r = f_11_1_r * w_x1 + f_21_1_r * w_x2
            b_1_r = f_12_1_r * w_x1 + f_22_1_r * w_x2
            F_1_r = a_1_r * w_y1 + b_1_r * w_y2

            # Upper Z plane, real part
            z_2_stride = z_2 * layer_size
            f_11_2_r = data_r[z_2_stride + y_1_stride + x_1]
            f_12_2_r = data_r[z_2_stride + y_2_stride + x_1]
            f_21_2_r = data_r[z_2_stride + y_1_stride + x_2]
            f_22_2_r = data_r[z_2_stride + y_2_stride + x_2]

            a_2_r = f_11_2_r * w_x1 + f_21_2_r * w_x2
            b_2_r = f_12_2_r * w_x1 + f_22_2_r * w_x2
            F_2_r = a_2_r * w_y1 + b_2_r * w_y2

            # Lower Z plane, imaginary part
            f_11_1_i = data_i[z_1_stride + y_1_stride + x_1]
            f_12_1_i = data_i[z_1_stride + y_2_stride + x_1]
            f_21_1_i = data_i[z_1_stride + y_1_stride + x_2]
            f_22_1_i = data_i[z_1_stride + y_2_stride + x_2]

            a_1_i = f_11_1_i * w_x1 + f_21_1_i * w_x2
            b_1_i = f_12_1_i * w_x1 + f_22_1_i * w_x2
            F_1_i = a_1_i * w_y1 + b_1_i * w_y2

            # Upper Z plane, imaginary part
            f_11_2_i = data_i[z_2_stride + y_1_stride + x_1]
            f_12_2_i = data_i[z_2_stride + y_2_stride + x_1]
            f_21_2_i = data_i[z_2_stride + y_1_stride + x_2]
            f_22_2_i = data_i[z_2_stride + y_2_stride + x_2]

            a_2_i = f_11_2_i * w_x1 + f_21_2_i * w_x2
            b_2_i = f_12_2_i * w_x1 + f_22_2_i * w_x2
            F_2_i = a_2_i * w_y1 + b_2_i * w_y2

            # Compute weights for Z dimension
            w_z1 = z_2 - (z[i] - origin_offset)
            w_z2 = (z[i] - origin_offset) - z_1

            # Calculate the real part of the result
            result_r[i] = F_1_r * w_z1 + F_2_r * w_z2

            # Calculate the imaginary part of the result
            result_i[i] = F_1_i * w_z1 + F_2_i * w_z2