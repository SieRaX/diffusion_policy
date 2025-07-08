if __name__ == "__main__":
    import sys
    import os
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
from typing import List, Dict, Optional
import numpy as np
from copy import deepcopy
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers.common import TimeLimit
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from robomimic.envs.env_robosuite import EnvRobosuite
from gymnasium_robotics.envs.adroit_hand.adroit_door import AdroitHandDoorEnv
from gymnasium_robotics.envs.adroit_hand.adroit_hammer import AdroitHandHammerEnv
from gymnasium_robotics.envs.adroit_hand.adroit_pen import AdroitHandPenEnv
from gymnasium_robotics.envs.adroit_hand.adroit_relocate import AdroitHandRelocateEnv


class D4RLLowdimWrapper(gym.Env):
    def __init__(self, 
        env: MujocoEnv,
        init_state: Optional[np.ndarray]=None,
        render_hw=(256,256)
        ):

        self.env = env.unwrapped # unwrap TimeLimit
        assert isinstance(self.env, (AdroitHandHammerEnv, AdroitHandRelocateEnv, AdroitHandDoorEnv, AdroitHandPenEnv)), "Environment must be one of Hammer, Relocate, Door, or Pen"
        self.env.sparse_reward = True
        mujoco_renderer = self.env.mujoco_renderer
        mujoco_renderer.width = render_hw[1]
        mujoco_renderer.height = render_hw[0]
        self.env.mujoco_renderer = mujoco_renderer
        self.render_hw = render_hw
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        
        # setup spaces
        self.action_space = Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            shape=self.env.action_space.shape,
            dtype=self.env.action_space.dtype
        )
        
        self.observation_space = Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype
        )

    def get_observation(self):
        obs = self.env._get_obs()
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self, **kwargs):
        if self.init_state is not None:
            # always reset to the same state
            # to be compatible with gym
            self.env.reset(options={"inital_state_dict": self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                self.env.reset(options={"inital_state_dict":
                    self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                self.env.reset()
                state = self.env.get_env_state()
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            self.env.reset(**kwargs)

        # return obs
        obs = self.env._get_obs()
        state = self.env.get_env_state()
        return obs, {"state_dict": state}
    
    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        
        reward = 0 if reward < 0 else 1
        
        info["state_dict"] = self.env.get_env_state()
        return raw_obs, reward, terminated, truncated, info
    
    def render(self, mode='rgb_array'):
        self.env.render_mode = mode
        img = self.env.render()
        # img = cv2.resize(img, self.render_hw)
        return img


def test():
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt
    
    from minari import MinariDataset
    
    # print(os.getcwd())
    dataset = MinariDataset(os.path.expanduser("~/Main/diffusion_policy/data/D4RL/hammer/expert-v2/data"))
    
    env = dataset.recover_environment()
    wrapper = D4RLLowdimWrapper(env, render_hw=(32,32))
    
    for _ in range(2):
        obs = wrapper.reset()
        # print(obs)
    
    img = wrapper.render()
    print(img.shape)
    plt.imsave('test_render.png', img)


if __name__ == "__main__":
    test()