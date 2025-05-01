import math
import numpy as np
from gym_duckietown.simulator import AGENT_SAFETY_RAD

# Threshold parameters for straight and turning modes
POSITION_THRESHOLD_CURVE = 0.01   
POSITION_THRESHOLD_STRAIGHT = 0.0006


VELOCITY_STRAIGHT = 0.7
GAIN_STRAIGHT = 2.5                 #Steering gain
FOLLOWING_DISTANCE_STRAIGHT = 7.0   # Look ahead distance in straight lines

VELOCITY_CURVE = 0.25       
GAIN_CURVE = 4.15                   # Steering gain
FOLLOWING_DISTANCE_CURVE = 0.325    # Look ahead distance in curves

D_GAIN = 5.0                        # Derivative gain
GAIN_DAGGER = 7.0           
VELOCITY_DAGGER = 0.8       
AGENT_SAFETY_GAIN = 1.15


class PurePursuitPolicy:
    """
    A Pure Pursuit controller class to act as an expert to the model
    ...
    Methods
    -------
    forward(images)
        makes a model forward pass on input images

    loss(*args)
        takes images and target action to compute the loss function used in optimization

    predict(observation)
        takes an observation image and predicts using env information the action
    """

    def __init__(
        self, env, ref_velocity=VELOCITY_CURVE, following_distance=FOLLOWING_DISTANCE_CURVE, 
        max_iterations=1000, DAgger=False
    ):
        """
        Parameters
        ----------
        env : gym environment
            duckietown simulation environment
        ref_velocity : float
            duckiebot maximum velocity
        following_distance : float
            distance used to follow the trajectory in pure pursuit
        max_iterations : int
            maximum number of iterations for finding curve points
        DAgger : bool
            whether using this policy for DAgger training
        """
        self.env = env
        self.max_iterations = max_iterations
        self.DAgger = DAgger

        self.following_distance = following_distance
        self.ref_velocity = ref_velocity
        self.P_gain = GAIN_CURVE
        self.D_gain = D_GAIN
        
        self.prev_err = 0

    def predict(self, observation):
        """
        Parameters
        ----------
        observation : image
            image of current observation from simulator
        Returns
        -------
        action: list
            action having velocity and omega of current observation
        """
        # Check for objects in front to avoid collisions
        current_world_objects = self.env.objects
        velocity_slow_down = 1
        for obj in current_world_objects:
            if not obj.static and obj.kind == "duckiebot":
                collision_penalty = abs(
                    obj.proximity(self.env.cur_pos, AGENT_SAFETY_RAD * AGENT_SAFETY_GAIN)
                )
                if collision_penalty > 0:
                    velocity_slow_down = collision_penalty
                    break

        # Get closest point on curve
        closest_point, closest_tangent = self.env.unwrapped.closest_curve_point(
            self.env.cur_pos, self.env.cur_angle
        )
        if closest_point is None:
            return [0.0, 0.0]  # Should return done in the environment

        # Find a point on the centerline of the road at the look ahead distance
        iterations = 0
        lookup_distance = self.following_distance
        curve_point = None
        while iterations < self.max_iterations:
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, tangent = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        if curve_point is None:
            return [0.0, 0.0]

        ### Determine if a curve is coming ###
        iterations = 0
        corner_lookup_distance = 0.375
        corner_curve_point = None
        while iterations < self.max_iterations:
            corner_follow_point = closest_point + closest_tangent * corner_lookup_distance
            corner_curve_point, corner_tangent = self.env.closest_curve_point(corner_follow_point, self.env.cur_angle)

            if corner_curve_point is not None:
                break
            iterations += 1
            corner_lookup_distance *= 0.5
        
        if corner_curve_point is None:
            return [0.0, 0.0]

        # Calculate the corner value to determine if we're in a curve
        posVec = corner_curve_point - closest_point
        upVec = np.array([0, 1, 0])
        rightVec = np.cross(corner_tangent, upVec)
        curve_value = np.dot(posVec, rightVec)
        abs_corner_value = np.absolute(curve_value)

        # Adjust parameters based on whether entering a curve or straight
        if (abs_corner_value > POSITION_THRESHOLD_CURVE) and (self.ref_velocity > VELOCITY_CURVE):
            self.P_gain = GAIN_CURVE
            self.ref_velocity = VELOCITY_CURVE
            self.following_distance = FOLLOWING_DISTANCE_CURVE
        
        if (abs_corner_value < POSITION_THRESHOLD_STRAIGHT) and (self.ref_velocity < VELOCITY_STRAIGHT): 
            self.P_gain = GAIN_STRAIGHT
            self.ref_velocity = VELOCITY_STRAIGHT
            self.following_distance = FOLLOWING_DISTANCE_STRAIGHT

        # Compute steering angle with PID control
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)
        
        right_vec = np.array([math.sin(self.env.cur_angle), 0, math.cos(self.env.cur_angle)])
        err = np.dot(right_vec, point_vec)
        derr = err - self.prev_err
        self.prev_err = err
        steering = self.P_gain * -err - self.D_gain * derr

        # Apply velocity reduction if obstacle is detected
        velocity = self.ref_velocity
        if velocity_slow_down < 0.2:
            velocity = 0
            steering = 0
        
        return [velocity, steering]
