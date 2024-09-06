import laspy
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt


class PointCloudVisualizer:
    def __init__(self, file_path):
        """
        Initializes the PointCloudVisualizer class.

        Args:
            file_path (str): The path to the .las file.
        """
        self.file_path = file_path
        self.point_cloud = None
        self.point_classes = None

    def load_las_file(self):
        """
        Loads the .las file and extracts the point cloud data and class information.
        """
        # Open the .las file
        las = laspy.read(self.file_path)

        # Extract point coordinates (X, Y, Z)
        points = np.vstack((las.x, las.y, las.z)).transpose()

        # Extract point classes
        point_classes = las.classification

        # Store the points and classes
        self.point_cloud = points
        self.point_classes = point_classes

    def get_color_map(self):
        """
        Generates a colormap for each point class.

        Returns:
            dict: A dictionary with colors for each class.
        """
        # Unique classes
        unique_classes = np.unique(self.point_classes)

        # Generate a colormap (using matplotlib's 'tab20' for distinct colors)
        cmap = plt.get_cmap('tab20', len(unique_classes))

        # Create a dictionary to map each class to a color
        class_color_map = {cls: cmap(i)[:3] for i, cls in enumerate(unique_classes)}

        return class_color_map

    def visualize(self):
        """
        Visualizes the point cloud in an interactive mode using Plotly with color coding for point classes.
        """
        if self.point_cloud is None or self.point_classes is None:
            raise ValueError("No point cloud loaded. Please run `load_las_file()` first.")

        # Get color map for the point cloud based on classes
        class_color_map = self.get_color_map()

        # Create a Plotly figure
        fig = go.Figure()

        # Plot each class separately to have separate legend entries
        unique_classes = np.unique(self.point_classes)
        for cls in unique_classes:
            # Get points of the current class
            indices = np.where(self.point_classes == cls)
            points_class = self.point_cloud[indices]

            # Get color for the current class
            color_rgb = class_color_map[cls]
            color_hex = 'rgb({},{},{})'.format(int(color_rgb[0] * 255), int(color_rgb[1] * 255), int(color_rgb[2] * 255))

            # Add a 3D scatter plot trace for the current class
            fig.add_trace(go.Scatter3d(
                x=points_class[:, 0],
                y=points_class[:, 1],
                z=points_class[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=color_hex,
                    opacity=0.8
                ),
                name=f"Class {cls}"  # Legend entry for the class
            ))

        # Update layout for a better 3D view
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            margin=dict(l=0, r=0, b=0, t=0),  # No margin
            showlegend=True  # Show the legend
        )

        # Show the plot
        pio.show(fig)


# Example Usage:
if __name__ == "__main__":
    # Code to execute when the script is run directly

    las_file_path = ""

    visualizer = PointCloudVisualizer(las_file_path)

    # Load the .las file
    visualizer.load_las_file()

    # Visualize the point cloud with class-specific colors and legend
    visualizer.visualize()
