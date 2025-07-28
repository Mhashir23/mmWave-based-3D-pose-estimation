import numpy as np
import os

def prepare_data_for_cnn(frame_files, ground_truth_file, m, max_points=108):
    feature_maps = []
    labels = []

    ground_truth = np.load(ground_truth_file)

    for frame_index in range(len(frame_files)):
        fused_points = []
        for i in range(-m, m + 1):
            current_index = frame_index + i
            if 0 <= current_index < len(frame_files):
                frame_data = np.fromfile(frame_files[current_index], dtype=np.float64).reshape(-1, 5)
                fused_points.append(frame_data[:, :5])

        fused_points = np.vstack(fused_points)

        if len(fused_points) > max_points:
            fused_points = fused_points[:max_points]
        else:
            padding = max_points - len(fused_points)
            fused_points = np.pad(fused_points, ((0, padding), (0, 0)), mode='constant')

        try:
            feature_map = fused_points.reshape(9, 12, 5)
            feature_maps.append(feature_map)
            labels.append(ground_truth[frame_index])
        except ValueError as e:
            print(f"Skipping frame {frame_index} due to shape mismatch: {e}")

    return np.array(feature_maps), np.array(labels)

root_folder = 'DB_Coursework'
m = 2
max_points = 108
all_feature_maps = []
all_labels = []

subjects = [f'S{i:02}' for i in range(1, 11)]
actions = [f'A{i:02}' for i in range(1, 28)]

for subject in subjects:
    for action in actions:
        folder_path = f"{root_folder}/{subject}/{action}"
        mmwave_folder = os.path.join(folder_path, 'mmWave')
        ground_truth_file = os.path.join(folder_path, 'ground_truth.npy')

        if os.path.exists(mmwave_folder) and os.path.exists(ground_truth_file):
            frame_files = sorted([os.path.join(mmwave_folder, f) for f in os.listdir(mmwave_folder) if f.endswith('.bin')])
            fm, lbl = prepare_data_for_cnn(frame_files, ground_truth_file, m, max_points=max_points)
            all_feature_maps.extend(fm)
            all_labels.extend(lbl)

# Convert to arrays
all_feature_maps = np.array(all_feature_maps)
all_labels = np.array(all_labels)

print(f"Feature maps shape: {all_feature_maps.shape}")  # Debugging
print(f"Labels shape: {all_labels.shape}")  # Debugging

# Save the data
np.save('train_feature_maps.npy', all_feature_maps)
np.save('train_labels.npy', all_labels)
