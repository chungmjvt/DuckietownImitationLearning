import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import argparse

# Import models from train.py
from train import create_imitation_cnn_with_ic, create_imitation_ncp, create_imitation_ct_rnn

# Load expert data
def load_expert_data(filename="imitation_data.npz"):
    if not os.path.exists(f"data/{filename}"):
        raise FileNotFoundError(f"File {filename} not found in data directory.")
    data = np.load(f"data/{filename}")
    return data['observations'], data['actions']

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['cnn', 'ncp'], default='cnn', help="Model type: 'cnn' or 'ncp'")
    args = parser.parse_args()

    observations, actions = load_expert_data()

    # Select and create model
    if args.model == 'cnn':
        model = create_imitation_cnn_with_ic()
        weight_file = 'imitation_cnn_weights.h5'
    elif args.model == 'ncp':
        model = create_imitation_ncp()
        weight_file = 'imitation_ncp_weights.h5'
    elif args.model == 'ctrnn':
        model = create_imitation_ct_rnn()
        weight_file = 'imitation_ct_rnn_weights.h5'

    # Load weights
    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"Weight file {weight_file} not found. Please train and save the model first.")
    model.load_weights(weight_file)
    print(f"Loaded weights from '{weight_file}'")

    # Predict
    predictions = model.predict(observations)

    # Metrics
    mae = mean_absolute_error(actions, predictions)
    r2 = r2_score(actions, predictions)
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # Separate component errors
    steering_mae = mean_absolute_error(actions[:, 0], predictions[:, 0])
    speed_mae = mean_absolute_error(actions[:, 1], predictions[:, 1])
    print(f"Steering MAE: {steering_mae:.4f}")
    print(f"Speed MAE: {speed_mae:.4f}")

    # Plot results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(actions[:, 0], label='True Steering')
    plt.plot(predictions[:, 0], label='Pred Steering')
    plt.title("Steering Comparison")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(actions[:, 1], label='True Speed')
    plt.plot(predictions[:, 1], label='Pred Speed')
    plt.title("Speed Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()

# to run: python evaluate_model.py --model cnn
# python evaluate_model.py --model ncp
# python evaluate_model.py --model ctrnn

