

import laspy
import pyvista as pv
import numpy as np


class PointCloudVisualizer:
    def __init__(self, file_path):
        """
        Initializes the PointCloudVisualizer class.

        Args:
            file_path (str): The path to the .las file.
        """
        self.file_path = file_path
        self.point_cloud = None

    def load_las_file(self):
        """
        Loads the .las file and extracts the point cloud data.
        """
        # Open the .las file
        las = laspy.read(self.file_path)

        # Extract point coordinates (X, Y, Z)
        points = np.vstack((las.x, las.y, las.z)).transpose()

        # Store the points
        self.point_cloud = points

    def visualize(self):
        """
        Visualizes the point cloud in an interactive mode.
        """
        if self.point_cloud is None:
            raise ValueError("No point cloud loaded. Please run `load_las_file()` first.")

        # Create a PyVista point cloud object
        cloud = pv.PolyData(self.point_cloud)

        # Create a plotter object
        plotter = pv.Plotter()

        # Add point cloud to the plotter
        plotter.add_points(cloud, color="white", point_size=2, render_points_as_spheres=True)

        # Set background color
        plotter.set_background("black")

        # Show the plot interactively
        plotter.show()


# Usage example
if __name__ == "__main__":
    # Code to execute when the script is run directly
    las_file_path = ""
    print(las_file_path)

    visualizer = PointCloudVisualizer(las_file_path)

    # Load the .las file
    visualizer.load_las_file()

    # Visualize the point cloud
    visualizer.visualize()

