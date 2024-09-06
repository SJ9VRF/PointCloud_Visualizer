import laspy
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from sklearn.svm import SVC


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

    def train_svm(self, class_a, class_b):
        """
        Trains a linear SVM to separate two classes and returns the plane parameters.

        Args:
            class_a, class_b (int): The class labels for the two classes to be separated.

        Returns:
            tuple: Coefficients (a, b, c, d) for the separating plane.
        """
        # Extract points belonging to Class A and Class B
        mask = np.isin(self.point_classes, [class_a, class_b])
        points_filtered = self.point_cloud[mask]
        classes_filtered = self.point_classes[mask]

        # Labels: Class A as 0, Class B as 1
        labels = np.where(classes_filtered == class_a, 0, 1)

        # Train a linear SVM classifier
        svm = SVC(kernel='linear')
        svm.fit(points_filtered, labels)

        # Get the separating hyperplane parameters
        # The normal vector to the plane is given by the coefficients of the SVM
        w = svm.coef_[0]
        a, b, c = w  # Coefficients for x, y, and z

        # The intercept gives us the constant d in the plane equation
        d = svm.intercept_[0]

        return a, b, c, d

    def plot_plane(self, a, b, c, d, x_range, y_range):
        """
        Plots a plane using the equation ax + by + cz + d = 0.

        Args:
            a, b, c, d (float): Coefficients of the plane equation.
            x_range (tuple): Range of x values for the plane.
            y_range (tuple): Range of y values for the plane.
        """
        # Create a meshgrid for the plane
        xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 10), np.linspace(y_range[0], y_range[1], 10))

        # Calculate the corresponding z values from the plane equation: z = (-d - ax - by) / c
        zz = (-d - a * xx - b * yy) / c

        # Return the meshgrid points for plotting
        return xx, yy, zz

    def visualize(self):
        """
        Visualizes the point cloud in an interactive mode using Plotly with color coding for point classes.
        Adds two SVM separating planes:
            - Plane 1: Separating Class 1 and Class 2
            - Plane 2: Separating Class 1 and Class 3
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
            color_hex = 'rgb({},{},{})'.format(int(color_rgb[0] * 255), int(color_rgb[1] * 255),
                                               int(color_rgb[2] * 255))

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

        # Get the ranges of x and y from the point cloud
        x_range = (np.min(self.point_cloud[:, 0]), np.max(self.point_cloud[:, 0]))
        y_range = (np.min(self.point_cloud[:, 1]), np.max(self.point_cloud[:, 1]))

        # Train SVM and get plane parameters for Class 1 and Class 2 (Plane 1)
        a1, b1, c1, d1 = self.train_svm(1, 2)
        # Get the plane meshgrid points
        xx1, yy1, zz1 = self.plot_plane(a1, b1, c1, d1, x_range, y_range)
        # Add the SVM plane for Class 1 and Class 2 as a surface to the plot
        fig.add_trace(go.Surface(
            x=xx1,
            y=yy1,
            z=zz1,
            opacity=0.5,
            colorscale='Viridis',
            showscale=False,
            name='Plane 1: Class 1 vs Class 2'
        ))

        # Train SVM and get plane parameters for Class 1 and Class 3 (Plane 2)
        a2, b2, c2, d2 = self.train_svm(1, 3)
        # Get the plane meshgrid points
        xx2, yy2, zz2 = self.plot_plane(a2, b2, c2, d2, x_range, y_range)
        # Add the SVM plane for Class 1 and Class 3 as a surface to the plot
        fig.add_trace(go.Surface(
            x=xx2,
            y=yy2,
            z=zz2,
            opacity=0.5,
            colorscale='Cividis',
            showscale=False,
            name='Plane 2: Class 1 vs Class 3'
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

    # Visualize the point cloud with two SVM planes (Class 1 vs Class 2 and Class 1 vs Class 3)
    visualizer.visualize()
