"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

        
# This process is needed to prevent the deadlock caused when try Image based env runner.
'''
# This is the anser from chat gpt
AsyncVectorEnv uses multiprocessing (fork-based on Unix)
It creates subprocesses that inherit the parent’s memory state.
But Mujoco’s internal C/OpenGL contexts cannot be safely inherited. When a child process attempts to initialize a new MjSim (via robosuite.make()), it often deadlocks or hangs in the call to mujoco_py.MjSim(...).

EGL or OpenGL context initialization happens per process
Robosuite tries to create an offscreen renderer (EGL) or OpenGL context. In forked processes, this often causes the context creation call to block forever.

egl_probe and GPU device selection make it worse
The block might happen at egl_probe.get_available_devices() (or inside EGL initialization), especially if each process tries to access the same GPU EGL context.
'''

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
