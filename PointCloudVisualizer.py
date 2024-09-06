import laspy
import open3d as o3d
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
        Visualizes the point cloud in an interactive mode using Open3D.
        """
        if self.point_cloud is None:
            raise ValueError("No point cloud loaded. Please run `load_las_file()` first.")

        # Create an Open3D PointCloud object
        o3d_cloud = o3d.geometry.PointCloud()

        # Convert the NumPy array of points to Open3D format
        o3d_cloud.points = o3d.utility.Vector3dVector(self.point_cloud)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([o3d_cloud])

if __name__ == "__main__":
    # Example Usage:
    # Create the visualizer object


    las_file_path = ".las"


    visualizer = PointCloudVisualizer(las_file_path)

    # Load the .las file
    visualizer.load_las_file()

    # Visualize the point cloud
    visualizer.visualize()
