import numpy as np
from copy import deepcopy
from diffusion_policy.env_runner.disturbance_generator.base_disturbance_generator import BaseDisturbanceGenerator
from diffusion_policy.common.pose_trajectory_interpolator import quat2matrix

class JumpingDisturbanceGenerator(BaseDisturbanceGenerator):
    def __init__(self, speed=0.03, max_number_of_disturbance=1, distance_threshold=0.04):
        super().__init__()
        self.speed = speed

        self._max_number_of_disturbance = max_number_of_disturbance
        self.number_of_disturbance = 0
        
        self.distance_threshold = distance_threshold
        
        self.slice = None
        
    def reset(self):
        self.number_of_disturbance = 0

    def generate_state_with_disturbance(self, env, **kwargs):
        if np.linalg.norm(env.get_observation()['object'][7:10]) < self.distance_threshold and self.number_of_disturbance < self._max_number_of_disturbance:
            direction = quat2matrix(env.get_observation()['robot0_eef_quat'])[0:2, 0]
            direction = direction/np.linalg.norm(direction)
            
            state = env.get_state()
            object_pos_quat = state['states'][self.slice]
            new_state = {k:v for k,v in deepcopy(state).items() if k != 'model'} # If model is copied the robot could not open the gripper.
            
            new_state['states'][self.slice] = object_pos_quat + np.array([*self.speed*direction, 0.0, 0.0, 0, 0, 0])
            self.number_of_disturbance += 1
            
            return new_state
        else:
            return super().generate_state_with_disturbance(env)