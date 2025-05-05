import gym
from gym import spaces
import numpy as np

from gym_duckietown.simulator import Simulator


class MotionBlurWrapper(Simulator):
    def __init__(self, env=None):
        Simulator.__init__(self)
        self.env = env
        self.frame_skip = 3
        self.env.delta_time = self.env.delta_time / self.frame_skip

    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1)
        # Actions could be a Python list
        action = np.array(action)
        motion_blur_window = []
        for _ in range(self.frame_skip):
            obs = self.env.render_obs()
            motion_blur_window.append(obs)
            self.env.update_physics(action)

        # Generate the current camera image

        obs = self.env.render_obs()
        motion_blur_window.append(obs)
        obs = np.average(motion_blur_window, axis=0, weights=[0.8, 0.15, 0.04, 0.01])

        misc = self.env.get_agent_info()

        d = self.env._compute_done_reward()
        misc["Simulator"]["msg"] = d.done_why

        return obs, d.reward, d.done, misc


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(60, 80, 3)):
        super(ResizeWrapper, self).__init__(env)
        # self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype,
        )
        self.shape = shape

    def observation(self, observation):
        import cv2
        return cv2.resize(observation, (self.shape[1], self.shape[0]))


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)

class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward


# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_

# Add this class at the end of the file
class GymCompatibilityWrapper(gym.Wrapper):
    """Converts between different Gym API versions."""
    
    def __init__(self, env):
        super(GymCompatibilityWrapper, self).__init__(env)
        
    def step(self, action):
        result = self.env.step(action)
        
        # Handle step API differences
        if len(result) == 4:  # Old API: obs, reward, done, info
            obs, reward, done, info = result
            return obs, reward, done, False, info  # Convert to new API
        else:  # New API: already 5 values
            return result
            
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        
        # Handle reset API differences
        if isinstance(result, tuple):
            if len(result) == 2:  # New API: obs, info
                return result
            elif len(result) > 2:  # Something returned too many values
                return result[0], result[1]  # Take first two values
        else:  # Old API: just obs
            return result, {}  # Convert to new API: obs, empty info