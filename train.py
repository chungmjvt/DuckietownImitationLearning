'''
TODO:
- Create keras pipeline
- Add NCP layers
- Try training
- Work a little on report
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np
import os

IMAGE_THRESHOLDING = False
USE_NORMALIZATION = True

# Load data
def load_expert_data(filename="imitation_data.npz"):
    if not os.path.exists(f"data/{filename}"):
        raise FileNotFoundError(f"File {filename} not found in data directory.")
    data = np.load(f"data/{filename}")
    observations = data['observations']
    actions = data['actions']
    print(f"Loaded expert data from data/{filename}.npz")
    return observations, actions

observations, actions = load_expert_data()
# Create TensorFlow dataset
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices((observations, actions))
dataset = dataset.shuffle(buffer_size=len(observations))
dataset = dataset.batch(batch_size)

# CNN model using Keras
def create_imitation_cnn_with_ic():
    model = Sequential([
        # First convolutional layer with IC (BatchNorm + SpatialDropout 0.01)
        Conv2D(2, kernel_size=5, strides=1, activation='relu', padding='same', 
               input_shape=(observations.shape[1], observations.shape[2], observations.shape[3])),
        BatchNormalization(),
        SpatialDropout2D(0.01),
        MaxPooling2D(pool_size=2, strides=2),
        
        # Second convolutional layer with IC (BatchNorm + SpatialDropout 0.05)
        Conv2D(12, kernel_size=5, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        SpatialDropout2D(0.05),
        MaxPooling2D(pool_size=2, strides=2),
        
        # Third convolutional layer with IC
        Conv2D(24, kernel_size=5, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        SpatialDropout2D(0.05),
        MaxPooling2D(pool_size=2, strides=2),
        
        # Fourth convolutional layer with IC
        Conv2D(36, kernel_size=5, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        SpatialDropout2D(0.05),
        MaxPooling2D(pool_size=2, strides=2),
        
        # Fifth convolutional layer with IC
        Conv2D(48, kernel_size=5, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        SpatialDropout2D(0.05),
        MaxPooling2D(pool_size=2, strides=2),
        Flatten(),

        # Fully connected layer with IC (BatchNorm + Dropout 0.05)
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.05),
        Dense(2)
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='mse'
    )
    
    return model

# Create and train the model
model = create_imitation_cnn_with_ic()
model.summary()

# Training
epochs = 5
history = model.fit(
    dataset,
    epochs=epochs,
    verbose=1
)

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.title('Model loss during training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Save the model
model.save('imitation_cnn.keras')
print("Model saved as 'imitation_cnn.keras'")
