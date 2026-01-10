from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces
from omegaconf import OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite

class RobomimicImageWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_hw=(256,256),
        render_obs_key='agentview_image',
        render_camera_name='agentview'
        ):

        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        
        # setup spaces
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

        self.time_step = 0
        self.hover_trigger = False

        if self.env.name == 'NutAssemblySquare':
            def is_it_failed():
                # r_reach, r_grasp, r_lift, r_hover = self.env.env.staged_rewards()
                # if r_hover > 0.55:
                #     self.hover_trigger = True 
                # return not r_grasp > 0 and self.time_step > 120 and not self.hover_trigger
                return False
        else:
            def is_it_failed():
                return False

        self.is_it_failed = is_it_failed


    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()
        
        self.render_cache = raw_obs[self.render_obs_key]

        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self, **kwargs):
        if 'seed' in kwargs:
            self._seed = kwargs['seed']
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # return obs
        obs = self.get_observation(raw_obs)
        state = self.env.get_state()['states']
        self.time_step = 0
        self.hover_trigger = False
        return obs, {"state_dict": state}
    
    def step(self, action):
        raw_obs, reward, _ , info = self.env.step(action)
        done = reward > 0 or self.is_it_failed()
        
        obs = self.get_observation(raw_obs)
        info["state_dict"] = self.env.get_state()['states']
        self.time_step += 1
        return obs, reward, done, False, info
    
    def render(self, mode='rgb_array'):
        # if self.render_cache is None:
        #     raise RuntimeError('Must run reset or step before render.')
        # img = np.moveaxis(self.render_cache, 0, -1)
        # h, w = self.render_hw
        # # Resize to (h, w) using numpy (nearest neighbor)
        # orig_h, orig_w, c = img.shape
        # if (orig_h, orig_w) != (h, w):
        #     y_idx = (np.linspace(0, orig_h - 1, h)).astype(np.int32)
        #     x_idx = (np.linspace(0, orig_w - 1, w)).astype(np.int32)
        #     img = img[y_idx,:,:]
        #     img = img[:,x_idx,:]
        # img = (img * 255).astype(np.uint8)
        # return img
        h, w = self.render_hw
        return self.env.render(mode=mode, 
            height=h, width=w, 
            camera_name=self.render_camera_name)


if __name__ == '__main__':
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('diffusion_policy/config/task/lift_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg['shape_meta']


    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = os.path.expanduser('data/robomimic/datasets/lift/ph/image.hdf5')
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=True, 
    )

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    img = wrapper.render()
    plt.imshow(img)


    # states = list()
    # for _ in range(2):
    #     wrapper.seed(0)
    #     wrapper.reset()
    #     states.append(wrapper.env.get_state()['states'])
    # assert np.allclose(states[0], states[1])

    # img = wrapper.render()
    # plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])
