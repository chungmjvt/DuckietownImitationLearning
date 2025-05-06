import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from utils.env import launch_env
from utils.wrappers import *
import cv2
from ncps.tf import LTC
from ncps.wirings import AutoNCP
from train import create_imitation_ncp, create_imitation_cnn_with_ic
import os

def _enjoy():
    # model = create_imitation_ncp()
    #load weights
    # model.load_weights("imitation_cnn_ncp.h5")
    model = create_imitation_cnn_with_ic()
    model.load_weights("imitation_cnn_ic.h5")
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False

    env = launch_env()
    env = GymCompatibilityWrapper(env)
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ActionWrapper(env)

    obs, _ = env.reset()
    observations = []
    actions = []
    
    for steps in range(0, 400):
        # Process the observation for the model
        obs_tensor = np.expand_dims(obs, axis=0)
        
        # Get the action from the model
        action = np.array([0.0, 0.0])
        action_raw = model.predict(obs_tensor, verbose=1)[0]
        action[0] = action_raw[0]
        action[1] = action_raw[1]
        # Apply the action to the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        observations.append(obs)
        actions.append(action)
        env.render()

        if done:
            if reward < 0:
                print("*** FAILED ***")
                time.sleep(0.7)

            obs = env.reset()
            env.render()

    create_expert_video(np.array(observations), np.array(actions))
    env.close()

def create_expert_video(observations, actions, filename="test_run_ncp.mp4"):
    """Create a video from the collected observations with action overlay."""
    print("Creating expert demonstration video...")
    
    # Get image dimensions and prepare video writer
    frame_count, height, width, channels = observations.shape
    sample_frame = observations[0]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    video = cv2.VideoWriter(f"videos/{filename}", fourcc, 10.0, (width, height))
    
    # Write each frame to video with action information overlay
    for i in range(len(observations)):
        # Get and prepare the frame
        frame = observations[i]
        
        # Convert to BGR if it's RGB
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame = (frame * 255).astype(np.uint8)
            
        if channels == 3:  # If RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Debug text for action overlay
        # action_text = f"Steering: {actions[i][1]:.3f}"
        # cv2.putText(frame, action_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
        #             0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Write the frame
        video.write(frame)
    
    # Release the video writer
    video.release()
    print(f"Video saved to videos/{filename}")

if __name__ == "__main__":
    _enjoy()

