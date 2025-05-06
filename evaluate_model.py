import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, SpatialDropout2D
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# Load expert data
def load_expert_data(filename="imitation_data.npz"):
    if not os.path.exists(f"data/{filename}"):
        raise FileNotFoundError(f"File {filename} not found in data directory.")
    data = np.load(f"data/{filename}")
    return data['observations'], data['actions']

# Define the same architecture used during training
def create_model(input_shape):
    model = Sequential([
        Conv2D(2, kernel_size=5, strides=1, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        SpatialDropout2D(0.01),
        MaxPooling2D(pool_size=2, strides=2),

        Conv2D(12, kernel_size=5, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        SpatialDropout2D(0.05),
        MaxPooling2D(pool_size=2, strides=2),

        Conv2D(24, kernel_size=5, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        SpatialDropout2D(0.05),
        MaxPooling2D(pool_size=2, strides=2),

        Conv2D(36, kernel_size=5, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        SpatialDropout2D(0.05),
        MaxPooling2D(pool_size=2, strides=2),

        Conv2D(48, kernel_size=5, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        SpatialDropout2D(0.05),
        MaxPooling2D(pool_size=2, strides=2),

        Flatten(),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.05),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Main
if __name__ == "__main__":
    observations, actions = load_expert_data()
    model = create_model(input_shape=(observations.shape[1], observations.shape[2], observations.shape[3]))
    model.load_weights("imitation_cnn_ic.h5")

    predictions = model.predict(observations)

    # Metrics
    mae = mean_absolute_error(actions, predictions)
    r2 = r2_score(actions, predictions)
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # Separate component errors
    steering_mae = mean_absolute_error(actions[:,0], predictions[:,0])
    speed_mae = mean_absolute_error(actions[:,1], predictions[:,1])
    print(f"Steering MAE: {steering_mae:.4f}")
    print(f"Speed MAE: {speed_mae:.4f}")

    # Plot
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
