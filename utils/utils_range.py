import laspy
import numpy as np


def get_xyz_ranges(file_path):
    """
    Reads a .las file and returns the range of X, Y, and Z values.

    Args:
        file_path (str): The path to the .las file.

    Returns:
        dict: A dictionary containing the min and max values for X, Y, and Z.
    """
    # Open the .las file
    las = laspy.read(file_path)

    # Extract the X, Y, and Z coordinates
    x = las.x
    y = las.y
    z = las.z

    # Calculate the min and max for each coordinate
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    # Return the ranges as a dictionary
    return {
        'x_range': (x_min, x_max),
        'y_range': (y_min, y_max),
        'z_range': (z_min, z_max)
    }


# Example Usage:
if __name__ == "__main__":

    las_file_path = ""


    ranges = get_xyz_ranges(las_file_path)
    print(f"X Range: {ranges['x_range']}")
    print(f"Y Range: {ranges['y_range']}")
    print(f"Z Range: {ranges['z_range']}")
