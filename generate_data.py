import time
import random
import argparse
import math
import json
from functools import reduce
import operator
import numpy as np
import matplotlib.pyplot as plt
from utils.env import launch_env
from utils.wrappers import *
# from utils.teacher import PurePursuitExpert
from utils.pure_pursuit_policy import PurePursuitPolicy
import cv2
import os

def _train(args):
    env = launch_env()
    env = GymCompatibilityWrapper(env)
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ActionWrapper(env)
    print("Initialized Wrappers")
    env.reset()
    env.render()
    time.sleep(0.5)
    observation_shape = (None,) + env.observation_space.shape
    action_shape = (None,) + env.action_space.shape

    # expert = PurePursuitExpert(env=env)
    max_velocity = 0.7
    expert = PurePursuitPolicy(env=env, ref_velocity=max_velocity)
    observations = []
    actions = []

    for episode in range(0, args.episodes):
        print("Starting episode", episode)
        for steps in range(0, args.steps):
            # use our 'expert' to predict the next action.
            action = expert.predict(None)
            observation, reward, terminated, done, info = env.step(action)
            observations.append(observation)
            actions.append(action)
            env.render()
        env.reset()
    
    actions = np.array(actions)
    observations = np.array(observations)

    if not os.path.exists("videos"):
        os.makedirs("videos")
    create_expert_video(observations, actions)
    save_expert_data(observations, actions)
    # load_observations, load_actions = load_expert_data()
    env.close()

def load_expert_data(filename="imitation_data.npz"):
    if not os.path.exists(f"data/{filename}"):
        raise FileNotFoundError(f"File {filename} not found in data directory.")
    data = np.load(f"data/{filename}")
    observations = data['observations']
    actions = data['actions']
    print(f"Loaded expert data from data/{filename}.npz")
    return observations, actions

#@brief Save expert data to numpy file
def save_expert_data(observations, actions, filename="imitation_data.npz"):
    if not os.path.exists('data'):
        os.makedirs('data')

    np.savez(f"data/{filename}", observations=observations, actions=actions)
    print(f"Expert data saved to data/{filename}.npz")

def create_expert_video(observations, actions, filename="expert_demonstration.mp4"):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1234, type=int, help="Sets Gym, TF, and Numpy seeds")
    parser.add_argument("--episodes", default=20, type=int, help="Number of epsiodes for experts")
    parser.add_argument("--steps", default=150, type=int, help="Number of steps per episode")
    parser.add_argument("--batch-size", default=32, type=int, help="Training batch size")
    parser.add_argument("--epochs", default=1, type=int, help="Number of training epochs")
    parser.add_argument("--model-directory", default="models/", type=str, help="Where to save models")

    args = parser.parse_args()

    _train(args)