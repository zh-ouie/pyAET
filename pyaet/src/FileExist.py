import os

def FileExist(FileName):
    """
    Check if a file exists.

    Args:
        FileName (str): The file path to check for existence.

    Returns:
        bool: True if the file exists, False otherwise.

    Notes:
        This function checks whether a file exists at the given file path.
    """
    return os.path.isfile(FileName)
