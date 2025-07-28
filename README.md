# mmWave-based 3D Pose Estimation

This project implements a deep learning pipeline for 3D human pose estimation using mmWave radar data. The code includes data preparation, model training (with hyperparameter tuning), evaluation, and visualization.

## Project Structure
- `prepare.py`: Prepares and processes the raw data into feature maps and labels.
- `cnn.py`, `cnn2.py`, `cnn3.py`, `cnn-tuned.py`: Different versions of the CNN model for pose estimation. `cnn-tuned.py` includes hyperparameter tuning with Keras Tuner.
- `testmodel.py`: Script for testing trained models.
- `visualise.py`: Visualizes predictions and results.
- `requirements.txt`: Python dependencies.
- `bayesian_tuning/`: Stores tuning results and checkpoints (do not push large files to GitHub).

## How to Run
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare data:**
   Ensure you have the required `.npy` data files (not included in the repo due to size). Run:
   ```bash
   python prepare.py
   ```
3. **Train the model:**
   ```bash
   python cnn-tuned.py
   ```
   This will perform hyperparameter tuning and save the best model.
4. **Test or visualize results:**
   ```bash
   python testmodel.py
   python visualise.py
   ```

## Notes
- Large data files (e.g., `.npy` files, model checkpoints) are excluded from version control. Store them separately.
- For best results, use a machine with a GPU and sufficient RAM.

## License
MIT License
