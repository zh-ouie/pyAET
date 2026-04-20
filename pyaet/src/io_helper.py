import os
import numpy as np
from scipy.io import loadmat
import pickle

def load_mat_save_npy(mat_file_path, output_file_path=None, data_key=None):
    """
    Load data from a .mat file and save it as a .npy file.

    Args:
        mat_file_path (str): The file path of the .mat file to be loaded.
        output_file_path (str): The file path of the .npy file to be saved. If None, save at the same path.
        data_key (str): the specific data_key in the .mat to be saved. If None, try finding the first valid data.

    Returns:
        A .npy file.
    """

    # Load data from the .mat file
    data = loadmat(mat_file_path)

    # Extract the data from the .mat file
    if data_key is None:
        # If not specified, find the first valid data
        for key, value in data.items():
            if not key.startswith("__"):
                break
        data_array = value
    else:
        data_array = data[data_key]

    # Create the .npy file path if not provided
    if output_file_path is None:
        base_name = os.path.splitext(os.path.basename(mat_file_path))[0]
        output_file_path = os.path.join(os.path.dirname(mat_file_path), f"{base_name}.npy")

    # Create the necessary directory structure for the .npy file
    dir_path = os.path.dirname(output_file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Save the data as a .npy file
    np.save(output_file_path, data_array)
    return


def load_mat_save_npz(mat_file_path, output_file_path=None, exclude_vars_starting_with='__'):
    """
    Load data from a .mat file and save all variables (except those starting with the specified prefix) as a single .npz file.

    Args:
        mat_file_path (str): The file path of the .mat file to be loaded.
        output_file_path (str, optional): The file path of the .npz file to be saved. If not provided, the .npz file will be saved in the same directory as the .mat file, with the same base name.
        exclude_vars_starting_with (str, optional): The prefix to use for excluding variables from the .npz file. Default is '__'.

    Returns:
        dict: A dictionary where the keys are the variable names from the .mat file (excluding those starting with the specified prefix), and the values are the corresponding numpy arrays.
    """

    # Load data from the .mat file
    data = loadmat(mat_file_path)

    # Create the .npz file path if not provided
    if output_file_path is None:
        base_name = os.path.splitext(os.path.basename(mat_file_path))[0]
        output_file_path = os.path.join(os.path.dirname(mat_file_path), f"{base_name}.npz")

        # Filter out variables starting with the specified prefix
        filtered_data = {k: v for k, v in data.items() if not k.startswith(exclude_vars_starting_with)}

        # Save the filtered data to the .npz file
        np.savez(output_file_path, **filtered_data)

    return filtered_data


def load_pickle_object(file_path):
    """
    Load a Python object from a file using the pickle module.

    Args:
        file_path (str): The file path to load the object from.

    Returns:
        object: The Python object loaded from the file.

    Raises:
        IOError: If there is an error reading the file.
        pickle.UnpicklingError: If there is an error unpickling the object.
    """
    with open(file_path, "rb") as file:
        obj = pickle.load(file)
    return obj


def read_mat_file(file_path):
    """
    Load data from a MATLAB .mat file and return the value associated with the first non-metadata key.

    Args:
        file_path (str): The file path to the .mat file.

    Returns:
        ndarray: The data associated with the first non-metadata key in the .mat file.

    Raises:
        Exception: If no valid non-metadata keys are found in the .mat file.
    """
    data = loadmat(file_path)
    data_keys = [key for key in data.keys() if not key.startswith('__')]
    if data_keys:
        return data[data_keys[0]]
    else:
        raise Exception(f'No valid data found in .mat file: {file_path}')

