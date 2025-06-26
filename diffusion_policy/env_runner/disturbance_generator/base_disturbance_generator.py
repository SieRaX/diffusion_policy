from copy import deepcopy

class BaseDisturbanceGenerator:
    def __init__(self):
        pass

    def generate_state_with_disturbance(self, env, **kwargs):
        state = env.get_state()
        new_state = {k:v for k,v in deepcopy(state).items() if k != 'model'}
        
        return new_state