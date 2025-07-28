import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
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

# Function to apply SOR (Statistical Outlier Removal), plot the fused point cloud, and connect ground truth points
def plot_fused_frames_with_ground_truth(frame_files, ground_truth_file, frame_index, m):
    # List to store all points from the fused frames
    fused_points = []

    # Load and concatenate points from M=2 previous, current, and next frames
    for i in range(-m, m + 1):
        current_index = frame_index + i
        if 0 <= current_index < len(frame_files):
            frame_data = np.fromfile(frame_files[current_index], dtype=np.float64)
            frame_data = frame_data.reshape(-1, 5)  # Assuming each point has (x, y, z, doppler, intensity)
            print(f"len of framed data: {len(frame_data)}")
            fused_points.append(frame_data[:, :3])  # Only keep (x, y, z)

    # Combine all points into a single numpy array
    fused_points = np.vstack(fused_points)
    print(f"len of fuse data {len(fused_points)}")

    # Load the ground truth data (17 joints, 3 coordinates per joint)
    ground_truth = np.load(ground_truth_file)

    # Extract the x, y, z coordinates of the 17 joints for the specific frame
    gt_X = ground_truth[frame_index, :, 0]  # Ground truth X for the 17 joints
    gt_Y = ground_truth[frame_index, :, 2]  # Ground truth Y for the 17 joints
    gt_Z = -ground_truth[frame_index, :, 1]  # Ground truth Z for the 17 joints

    # Create an Open3D PointCloud object for the fused mmWave data
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(fused_points)

    # Apply Statistical Outlier Removal (SOR)
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    cleaned_points = np.asarray(cl.points)

    # Create a figure for plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the cleaned mmWave signal data points (after SOR)
    ax.scatter(cleaned_points[:, 0], cleaned_points[:, 1], cleaned_points[:, 2], c='blue', s=1, label='Cleaned Fused mmWave Signal')

    # Plot the ground truth data points (17 joints)
    ax.scatter(gt_X, gt_Y, gt_Z, c='red', s=10, marker='o', label='Ground Truth (Joints)')

    # Connect ground truth points with lines
    for i in range(len(gt_X) - 1):  # Loop through all joints except the last
        ax.plot([gt_X[i], gt_X[i + 1]], 
                [gt_Y[i], gt_Y[i + 1]], 
                [gt_Z[i], gt_Z[i + 1]], 
                color='green', linewidth=1.5)

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Fused Scatter Plot - Frame {frame_index + 1}')

    # Set equal scaling for the axes
    set_axes_equal(ax)

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

# Folder name where the dataset is located (relative to the current working directory)
root_folder = 'DB_Coursework'

# Number of frames to fuse (M=2 means 5 frames: current, 2 before, 2 after)
m = 2

# Iterate through all subjects (S01 to S10)
for subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
    subject_folder = f'{root_folder}/{subject}'

    # Iterate through all A folders (A01 to A27)
    for a_folder in [f'A{i:02}' for i in range(1, 28)]:
        a_folder_path = f'{subject_folder}/{a_folder}'
        mmwave_folder = f'{a_folder_path}/mmWave'
        ground_truth_file = f'{a_folder_path}/ground_truth.npy'

        # Check if mmWave folder exists and ground truth file exists
        if os.path.exists(mmwave_folder) and os.path.exists(ground_truth_file):  
            # Get all .bin files (frames) in the mmWave folder
            bin_files = sorted([os.path.join(mmwave_folder, f) for f in os.listdir(mmwave_folder) if f.endswith('.bin')])

            # For each frame file, plot the corresponding fused scatter plot with ground truth and SOR
            for frame_index in range(len(bin_files)):
                plot_fused_frames_with_ground_truth(bin_files, ground_truth_file, frame_index, m)
