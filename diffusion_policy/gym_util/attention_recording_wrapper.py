import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill
from matplotlib import pyplot as plt
from PIL import Image

class AttentionRecordingWrapper(gym.Wrapper):
    def __init__(self, env, max_timesteps=70, max_attention=700.0):
        super().__init__(env)
        self.attention_pred_list = None
        self.sample_triggered_list = None
        self.timestep = 0
        self.max_timesteps = max_timesteps
        self.max_attention = max_attention

    def reset(self, **kwargs):
        self.attention_pred_list = None
        self.sample_triggered_list = None
        self.timestep = 0
        res = super().reset(**kwargs)
        return res
    
    def step(self, action):
        self.timestep += 1
        return super().step(action)

    def render(self, mode='rgb_array', **kwargs):

        assert self.attention_pred_list is not None and self.sample_triggered_list is not None, "attention_pred_list and sample_triggered_list must be set"

        if mode == 'rgb_array':
            fig, ax = plt.subplots(figsize=(4, 3))
            
            time_steps = np.arange(len(self.attention_pred_list))
            
            ax.set_xlim(0, self.max_timesteps)
            ax.set_ylim(-0.5, self.max_attention)
            ax.plot(time_steps, self.attention_pred_list, 'b-', linewidth=1)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Attention Pred')
            ax.grid(True)
            ax.scatter(np.arange(0, self.timestep), self.attention_pred_list[:self.timestep], color='red', s=30)

            trigger_index = np.where(self.sample_triggered_list[:self.timestep])[0]
            ax.scatter(trigger_index, 
                      np.zeros_like(trigger_index), 
                      color='green', s=30, marker='x')

            fig.canvas.draw()
            graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            graph_height, graph_width = graph_img.shape[:2]
            plt.close(fig)

            frame = self.env.render(mode, **kwargs)

            frame_height, frame_width = frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            
            new_height = graph_height
            new_width = int(new_height * aspect_ratio)

            frame_resized = np.array(Image.fromarray(frame).resize((new_width, new_height)))
            
            canvas = np.zeros((graph_height, graph_width+new_width, 3), dtype=np.uint8)
            canvas[:, :new_width, :] = frame_resized
            canvas[:, new_width:, :] = graph_img
            
            return canvas

        else:
            return super().render(mode, **kwargs)