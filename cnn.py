import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Conv2D, BatchNormalization, Dropout
import tensorflow as tf
import pandas as pd

# Load the prepared data
train_features = np.load('train_feature_maps.npy')  # Feature maps
train_labels = np.load('train_labels.npy')          # Ground truth labels
print(f"Original train_labels shape: {train_labels.shape}")  # Should be [80190, 17, 3] before flattening

train_labels = train_labels.reshape(train_labels.shape[0], -1)  # Flatten labels

# Split the data into training, validation, and testing sets
num_samples = len(train_features)
split_train = int(0.6 * num_samples)  # 70% training
split_val = int(0.8 * num_samples)   # 15% validation, 15% testing

x_train, y_train = train_features[:split_train], train_labels[:split_train]
x_val, y_val = train_features[split_train:split_val], train_labels[split_train:split_val]
x_test, y_test = train_features[split_val:], train_labels[split_val:]

# Define the CNN model
def define_cnn(input_shape, num_joints):
    inputs = Input(shape=input_shape)
    
    # Convolutional layers
    conv1 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.3)(conv1)
    conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv2 = Dropout(0.3)(conv2)
    
    # Batch normalization
    conv2 = BatchNormalization(momentum=0.95)(conv2)

    # Flatten and dense layers
    flat = Flatten()(conv2)
    dense1 = Dense(512, activation='relu')(flat)
    dense1 = BatchNormalization(momentum=0.95)(dense1)
    dense1 = Dropout(0.4)(dense1)

    # Output layer
    outputs = Dense(num_joints * 3, activation='linear')(dense1)  # 3D coordinates for each joint

    # Define the model
    model = Model(inputs, outputs)
    optimizer = Adam(learning_rate=0.001, beta_1=0.5)

    # Compile the model
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError()])
    
    return model

# Initialize the model
input_shape = x_train.shape[1:]  # (9, 12, 5) or other dimensions based on your preprocessing
num_joints = train_labels.shape[1] // 3  # Assuming labels are flattened
cnn_model = define_cnn(input_shape, num_joints)

# Train the model
batch_size = 128
epochs = 150

history = cnn_model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    verbose=1
)

# Evaluate the model on test data
score = cnn_model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {score[0]}, Test MAE: {score[1]}, Test RMSE: {score[2]}")

# Predict on the test set
predictions = cnn_model.predict(x_test)
# Metrics: Mean Absolute Error for each axis
mae_x = metrics.mean_absolute_error(y_test[:, 0::3], predictions[:, 0::3])
mae_y = metrics.mean_absolute_error(y_test[:, 1::3], predictions[:, 1::3])
mae_z = metrics.mean_absolute_error(y_test[:, 2::3], predictions[:, 2::3])
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

print(f"MAE for X: {mae_x:.4f}, Y: {mae_y:.4f}, Z: {mae_z:.4f}")
# Save joint-wise and total evaluation results to a CSV file
total_res = pd.DataFrame({
    'MAE for X': [mae_x],
    'MAE for Y': [mae_y],
    'MAE for Z': [mae_z],
    'RMSE': [rmse]
})
total_res.to_csv('cnn2res.csv', index=False)

# Visualization: Plot accuracy and loss
plt.figure(figsize=(12, 5))

# Plot Mean Absolute Error
plt.subplot(1, 2, 1)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('cnn2.png')
plt.show()
# Reshape predictions and ground truth back to [num_samples, num_joints, 3]
y_test_reshaped = y_test.reshape(-1, num_joints, 3)
predictions_reshaped = predictions.reshape(-1, num_joints, 3)



# Initialize lists to store MAE and RMSE for each joint
mae_per_joint = []
rmse_per_joint = []


# Evaluate each joint
for joint_idx in range(num_joints):
    # Extract ground truth and predictions for the current joint
    y_true_joint = y_test_reshaped[:, joint_idx, :]  # Shape: [num_samples, 3]
    y_pred_joint = predictions_reshaped[:, joint_idx, :]  # Shape: [num_samples, 3]
    
    # Compute MAE and RMSE for the current joint
    mae = metrics.mean_absolute_error(y_true_joint, y_pred_joint)
    rmse = np.sqrt(metrics.mean_squared_error(y_true_joint, y_pred_joint))
    
    # Append results to the lists
    mae_per_joint.append(mae)
    rmse_per_joint.append(rmse)
    
    # Accumulate total MAE and RMSE



total_mae = np.mean(mae_per_joint)
total_rmse = np.mean(rmse_per_joint)

# Append total results to the lists
mae_per_joint.append(total_mae)
rmse_per_joint.append(total_rmse)

# Save joint-wise and total evaluation results to a CSV file
eval_results = pd.DataFrame({
    'Joint': [f'Joint {i+1}' for i in range(num_joints)] + ['Total'],
    'MAE': mae_per_joint,
    'RMSE': rmse_per_joint
})

# Save results to CSV
eval_results.to_csv('joint_wise_evaluation_with_totals.csv', index=False)

# Display joint-wise evaluation including totals
print("Joint-wise Evaluation with Total MAE and RMSE:")
print(eval_results)

# Visualization
plt.figure(figsize=(12, 6))

# Plot MAE per joint
plt.subplot(1, 2, 1)
plt.bar(range(1, num_joints + 2), mae_per_joint, color='skyblue')  # num_joints + 2 for the Total entry
plt.title('Mean Absolute Error (MAE) per Joint')
plt.xlabel('Joint Index')
plt.ylabel('MAE')
plt.xticks(range(1, num_joints + 2), [f'Joint {i+1}' for i in range(num_joints)] + ['Total'], rotation=45)

# Plot RMSE per joint
plt.subplot(1, 2, 2)
plt.bar(range(1, num_joints + 2), rmse_per_joint, color='salmon')  # num_joints + 2 for the Total entry
plt.title('Root Mean Squared Error (RMSE) per Joint')
plt.xlabel('Joint Index')
plt.ylabel('RMSE')
plt.xticks(range(1, num_joints + 2), [f'Joint {i+1}' for i in range(num_joints)] + ['Total'], rotation=45)

plt.tight_layout()
plt.savefig('joint_wise_evaluation_with_totals.png')
plt.show()


# Save the model and results
cnn_model.save('mars_cnn_model2.h5')
np.save('test_predictions2.npy', predictions)
np.save('test_ground_truth2.npy', y_test)
