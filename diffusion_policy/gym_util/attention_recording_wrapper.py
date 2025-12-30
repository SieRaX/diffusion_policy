import gymnasium as gym
import numpy as np
from collections import defaultdict, deque
import dill
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from diffusion_policy.env.d4rl.d4rl_lowdim_wrapper import D4RLLowdimWrapper
from gymnasium_robotics.envs.adroit_hand.adroit_hammer import AdroitHandHammerEnv
from gymnasium_robotics.envs.adroit_hand.adroit_door import AdroitHandDoorEnv

def get_additional_info(env:gym.Env):
    
    if type(env) == D4RLLowdimWrapper:
        mujoco_env = env.env
        
        if type(mujoco_env) == AdroitHandHammerEnv:
            nail_pos = mujoco_env.data.site_xpos[mujoco_env.target_obj_site_id].ravel()
            goal_pos = mujoco_env.data.site_xpos[mujoco_env.goal_site_id].ravel()
            return {'nail_distance': [np.linalg.norm(nail_pos - goal_pos), 0.01]}

        elif type(mujoco_env) == AdroitHandDoorEnv:
            door_pos = mujoco_env.data.qpos[mujoco_env.door_hinge_addrs]
            return {'door_pos': [door_pos, 1.35]}

        return dict()
        
    else:
        return dict()

class AttentionRecordingWrapper(gym.Wrapper):
    def __init__(self, env, max_timesteps=70, max_attention=700.0):
        super().__init__(env)
        self.attention_pred_list = None
        self.sample_triggered_list = None
        self.timestep = 0
        self.max_timesteps = max_timesteps
        self.max_attention = max_attention
        self.additional_info = dict()

    def reset(self, **kwargs):
        self.attention_pred_list = None
        self.sample_triggered_list = None
        self.timestep = 0
        self.additional_info = dict()
        res = super().reset(**kwargs)
        
        additional_info = get_additional_info(self.env)
        
        for key, value in additional_info.items():
            self.additional_info[key] = list()        
        return res
    
    def step(self, action):
        self.timestep += 1
        
        additional_info = get_additional_info(self.env)
        
        for key, value in additional_info.items():
            self.additional_info[key].append(value)
                
        return super().step(action)
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def render(self, mode='rgb_array', **kwargs):

        if self.attention_pred_list is not None and self.sample_triggered_list is not None:
            pass

        else:
            self.attention_pred_list = np.zeros(self.max_timesteps)
            self.sample_triggered_list = np.zeros(self.max_timesteps)
            
        # assert len(self.additional_info) > 0, "additional_info must be set" # We don't need additional info assertion for rendering

        if mode == 'rgb_array':
            fig, ax = plt.subplots(figsize=(4, 3))
            
            time_steps = np.arange(len(self.attention_pred_list))
            
            ax.set_xlim(0, self.max_timesteps)
            ax.set_ylim(-0.05*self.max_attention, self.max_attention)
            ax.plot(time_steps, self.attention_pred_list, 'b-', linewidth=1)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Attention Pred')
            ax.grid(True)
            ax.scatter(np.arange(0, self.timestep), self.attention_pred_list[:self.timestep], color='red', s=30)

            trigger_index = np.where(self.sample_triggered_list[:self.timestep])[0]

            if len(trigger_index) == 1:
                horizon_length = len(self.attention_pred_list)
            else:
                horizon_length = len(self.attention_pred_list) - trigger_index[-1]

            # Draw horizon_length in upper right corner of the graph
            ax.annotate(
                f"horizon_length: {horizon_length}",
                xy=(1.0, 1.02),
                xycoords='axes fraction',
                fontsize=10,
                ha='right',
                va='bottom',
                fontweight='bold',
                color='black',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray")
            )

            self.attention_length_before = len(self.attention_pred_list)
            
            ax.scatter(trigger_index, 
                      np.zeros_like(trigger_index), 
                      color='green', s=50, marker='x')

            fig.canvas.draw()
            graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            graph_height, graph_width = graph_img.shape[:2]
            plt.close(fig)
            
            graph_imgs_additional_list = list()
            graph_widths_additional_list = list()
            for key, value_and_goal in self.additional_info.items():
                value_and_goal = np.array(value_and_goal)
                value = value_and_goal[:, 0]
                goal = value_and_goal[:, 1]
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.set_xlim(0, self.max_timesteps)
                ax.set_ylim(bottom=0.0, top=max(value)+0.01)
                ax.plot(np.arange(0, self.timestep), value, 'b-', linewidth=1)
                ax.plot(np.arange(0, self.timestep), np.ones_like(value)*goal[0], 'm--', linewidth=1)
                ax.set_xlabel('Time Step')
                ax.set_title(key)
                # ax.set_ylabel(key)
                ax.grid(True)
                fig.canvas.draw()
                
                graph_img_additional = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                graph_img_additional = graph_img_additional.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                graph_height, graph_width_additional = graph_img_additional.shape[:2]
                plt.close(fig)
                
                graph_imgs_additional_list.append(graph_img_additional)
                graph_widths_additional_list.append(graph_width_additional)

            frame = self.env.render(mode, **kwargs)

            frame_height, frame_width = frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            
            new_height = graph_height
            new_width = int(new_height * aspect_ratio)

            frame_resized = np.array(Image.fromarray(frame).resize((new_width, new_height)))
            
            total_graph_width = graph_width + sum(graph_widths_additional_list)
            canvas = np.zeros((graph_height+int(graph_height*0.2), total_graph_width+new_width, 3), dtype=np.uint8)
            canvas[:graph_height, :new_width, :] = frame_resized
            canvas[:graph_height, new_width:new_width+graph_width, :] = graph_img
            for i, graph_img_additional in enumerate(graph_imgs_additional_list):
                canvas[:graph_height, new_width+graph_width+sum(graph_widths_additional_list[:i]):new_width+graph_width+sum(graph_widths_additional_list[:i+1]), :] = graph_img_additional
            
            return canvas

        else:
            return super().render(mode, **kwargs)