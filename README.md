# Duckietown Imitation Learning

This repository contains data generation and implementation of immitation learning agent for Duckietown.

## Directory Structure

```
duckietown_imitation_learning/
└── utils/          # Utility functions for data generation
    └── env.py      # Gym environment setupu 
    └── teacher.py  # Agent/Algorithm for data generation
    └── wrapper.py  # Environment wrapper for Duckietown
└── train.py        # Training script for imitation learning
└── imitation.py    # Runs the imitation learning algorithm in the sim 
```

## Data Collection

### Pure Pursuit Data Collection

Collect data using pure pursuit controller. \
This is a basic controller that follows a path using a lookahead point.

### Data Collection from Demonstration
TODO: Collect data by driving the Duckiebot manually

**Controls:**
- Arrow keys: Drive the Duckiebot
- ENTER: Start/stop recording
- BACKSPACE or /: Reset the environment

## Data Format

The collected data is stored in memory to be used directly or it can be saved to a file. \

```python
{
    'observation': numpy.ndarray,  # The observation image
    'action': numpy.ndarray,       # The action taken [speed, steering angle]
}
```

Length of the data is number of steps * number of episodes.

## Data Preprocessing

The `utils/wrappers.py` is responsible for preprocessing the data. \

**Wrappers in order:**
-   GymCompatibilityWrapper: Matches the new gym API expected number returns from step and reset to old gym API
-   ResizeWrapper: Resizes the image to a smaller size [120, 160, 3]
-   NormalizeWrapper: Normalizes the image to [0, 1]
-   ImgWrapper: Adjust the image to the correct format for the neural network
-   ActionWrapper: Caps the velocity to 0.8 instead of 1.0. At max speed the agent can't turn efficiently
-   DtRewardWrapper: Not used in this implementation.


## Imitation Learning Implementation

TODO:
-   Improve the teacher agent