import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import defaultdict, deque
import dill
from diffusion_policy.env_runner.disturbance_generator.jumping_disturbance_generator import BaseDisturbanceGenerator

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])

def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result

def aggregate(data, method='max'):
    if method == 'max':
        # equivalent to any
        return np.max(data)
    elif method == 'min':
        # equivalent to all
        return np.min(data)
    elif method == 'mean':
        return np.mean(data)
    elif method == 'sum':
        return np.sum(data)
    else:
        raise NotImplementedError()

def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max'
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps

        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
    
    def reset(self, **kwargs):
        """Resets the environment using kwargs."""
        obs, info = super().reset(**kwargs)

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.terminated = list()
        self.truncated = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))
        self._add_info(info)

        obs = self._get_obs(self.n_obs_steps)
        info = dict_take_last_n(self.info, 1)
        return obs, info

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        self.env.action_seq = action
        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            observation, reward, terminated, truncated, info = super().step(act)
            done = terminated or truncated

            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
                terminated = True
            self.done.append(done)
            self.terminated.append(terminated)
            self.truncated.append(truncated)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        terminated = aggregate(self.terminated, 'max')
        truncated = aggregate(self.truncated, 'max')
        done = terminated or truncated
        info = dict_take_last_n(self.info, self.n_obs_steps)
        
        return observation, reward, terminated, truncated, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info, to_env=False):
        for key, value in info.items():
            self.info[key].append(value)
        if to_env:
            for key, value in info.items():
                assert key == 'action_pred'
                self.env.info[key] = value
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
    
    def seed(self, seed=None):
        return self.env.seed(seed)

class SubMultiStepWrapperwithDisturbance(MultiStepWrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max',
            disturbance_generator=BaseDisturbanceGenerator(),
        ):
        super().__init__(env, n_obs_steps, n_action_steps, max_episode_steps, reward_agg_method)
        self.disturbance_generator = disturbance_generator
        self.horizon_idx = None
        self.horizon_idx_list = list()
        self.total_steps = 0
        self.done = list()
        self.attention_pred_list = list()
        self.sample_triggered_list = list()
        self.complete = False
        self.c_att = None

    def set_invalid_env(self, is_it_invalid):
        if is_it_invalid:
            self.reward = [-1.0] * self.max_episode_steps

    def set_complete(self, complete):
        self.complete = complete
    
    def reset(self, **kwargs):
        if self.complete:
            # This is for the capability of env runner.
            # If self.complete is True, the env runner will not run at all.
            obs = self._get_obs(self.n_obs_steps)
            info = dict_take_last_n(self.info, 1)
            return obs, info
        else:
            if self.disturbance_generator is not None:
                self.disturbance_generator.reset()
            self.horizon_idx = None
            self.horizon_idx_list = list()
            self.total_steps = 0
            self.done = list()
            self.attention_pred_list = list()
            self.sample_triggered_list = list()
            self.c_att = None

            res = super().reset(**kwargs)
            return res
    
    def register_c_att(self, c_att):
        if not self.complete:
            self.c_att = c_att
        
    def register_horizon_idx(self, horizon_idx):
        if not self.complete:
            self.horizon_idx = horizon_idx
            self.horizon_idx_list.append(horizon_idx)
    
    def register_attention_pred(self, attention_pred):
        if not self.complete:
            self.attention_pred_list.append(attention_pred)
            sample_triggered = np.zeros(attention_pred.shape[0], dtype=np.bool)
            sample_triggered[0]=True
            self.sample_triggered_list.append(sample_triggered)
    
    def deregister_horizon_idx(self):
        if not self.complete:
            self.horizon_idx = None

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        if self.complete:
            # Do not do anything when complete is True.
            # Just give them the last observation.
            observation = self._get_obs(self.n_obs_steps)
            reward = aggregate(self.reward, self.reward_agg_method)
            terminated = aggregate(self.terminated, 'max')
            truncated = aggregate(self.truncated, 'max')
            done = terminated or truncated
            info = dict_take_last_n(self.info, self.n_obs_steps)
            return observation, reward, terminated, truncated, info

        else:
            assert self.horizon_idx is not None, f"horizon_idx is not registered"
            self.env.action_seq = action
            
            self.env.env.attention_pred_list = np.concatenate(self.attention_pred_list, axis=0)
            self.env.env.sample_triggered_list = np.concatenate(self.sample_triggered_list, axis=0)

            for act in action[:self.horizon_idx]:
                if len(self.done) > 0 and self.done[-1]:
                    # termination
                    break
                
                observation, reward, terminated, truncated, info = gym.Wrapper.step(self, act)
                self.total_steps += 1
                done_el = terminated or truncated

                if self.disturbance_generator is not None:
                    new_state = self.disturbance_generator.generate_state_with_disturbance(self.env.env.env.env)
                    self.env.env.env.env.reset_to(new_state)

                self.obs.append(observation)
                self.reward.append(reward)
                if (self.max_episode_steps is not None) \
                    and (len(self.reward) >= self.max_episode_steps):
                    # truncation
                    done_el = True
                    terminated = True
                    truncated = True

                #     self.done.append(done_el)
                #     self.terminated.append(terminated)
                #     self.truncated.append(truncated)
                #     self._add_info(info)

                #     break

                # else:
                #     self.done.append(done_el)
                #     self.terminated.append(terminated)
                #     self.truncated.append(truncated)
                #     self._add_info(info)

                self.done.append(done_el)
                self.terminated.append(terminated)
                self.truncated.append(truncated)
                self._add_info(info)

            observation = self._get_obs(self.n_obs_steps)
            reward = aggregate(self.reward, self.reward_agg_method)
            terminated = aggregate(self.terminated, 'max')
            truncated = aggregate(self.truncated, 'max')
            # done = terminated or truncated
            info = dict_take_last_n(self.info, self.n_obs_steps)
            return observation, reward, terminated, truncated, info