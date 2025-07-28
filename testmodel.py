import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to ensure equal scaling for 3D plots
def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = np.abs(limits[:, 1] - limits[:, 0])
    max_span = spans.max()
    centers = np.mean(limits, axis=1)
    new_limits = np.array([centers - max_span / 2, centers + max_span / 2]).T
    ax.set_xlim3d(new_limits[0])
    ax.set_ylim3d(new_limits[1])
    ax.set_zlim3d(new_limits[2])

# Function to visualize predictions and ground truth
def visualize_predictions(predictions_file, ground_truth_file, num_samples=5):
    """
    Visualize predictions and ground truth joint positions in 3D scatter plots with connections.

    Args:
    - predictions_file (str): Path to the file containing predicted joint coordinates (npy file).
    - ground_truth_file (str): Path to the file containing ground truth joint coordinates (npy file).
    - num_samples (int): Number of samples to visualize.
    """
    # Load predictions and ground truth
    predictions = np.load(predictions_file)  # Shape: [num_samples, num_joints * 3]
    ground_truth = np.load(ground_truth_file)  # Shape: [num_samples, num_joints * 3]

    # Determine number of joints
    num_joints = predictions.shape[1] // 3

    # Reshape into [num_samples, num_joints, 3]
    predictions_reshaped = predictions.reshape(-1, num_joints, 3)
    ground_truth_reshaped = ground_truth.reshape(-1, num_joints, 3)

    # Limit to the specified number of samples
    num_samples = min(num_samples, predictions_reshaped.shape[0])

    for i in range(num_samples):
        plt.figure(figsize=(12, 6))

        # Ground truth
        ax = plt.subplot(1, 2, 1, projection='3d')
        ax.scatter(
            ground_truth_reshaped[i, :, 0], 
            ground_truth_reshaped[i, :, 2], 
            -ground_truth_reshaped[i, :, 1], 
            c='blue', label='Ground Truth'
        )
        # Connect ground truth joints with lines
        for j in range(num_joints - 1):
            ax.plot(
                [ground_truth_reshaped[i, j, 0], ground_truth_reshaped[i, j + 1, 0]],
                [ground_truth_reshaped[i, j, 2], ground_truth_reshaped[i, j + 1, 2]],
                [-ground_truth_reshaped[i, j, 1], -ground_truth_reshaped[i, j + 1, 1]],
                color='green', linewidth=1.5
            )
        ax.set_title('Ground Truth')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        set_axes_equal(ax)
        ax.legend()

        # Predictions
        ax = plt.subplot(1, 2, 2, projection='3d')
        ax.scatter(
            predictions_reshaped[i, :, 0], 
            predictions_reshaped[i, :, 2], 
            -predictions_reshaped[i, :, 1], 
            c='red', label='Prediction'
        )
        # Connect predicted joints with lines
        for j in range(num_joints - 1):
            ax.plot(
                [predictions_reshaped[i, j, 0], predictions_reshaped[i, j + 1, 0]],
                [predictions_reshaped[i, j, 2], predictions_reshaped[i, j + 1, 2]],
                [-predictions_reshaped[i, j, 1], -predictions_reshaped[i, j + 1, 1]],
                color='orange', linewidth=1.5
            )
        ax.set_title('Prediction')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        set_axes_equal(ax)
        ax.legend()

        plt.tight_layout()
        plt.show()

# Example usage
visualize_predictions('test_predictions5.npy', 'test_ground_truth5.npy', num_samples=50)
