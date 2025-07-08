import sys
import numpy as np
import time

from gymnasium.vector import AsyncVectorEnv as AsyncVectorEnvBase
from gymnasium.vector.async_vector_env import AsyncState
from gymnasium.error import (
    AlreadyPendingCallError,
    NoAsyncCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
)
import multiprocessing as mp
from enum import Enum
from gymnasium.vector.utils import (
    create_shared_memory,
    create_empty_array,
    write_to_shared_memory,
    read_from_shared_memory,
    concatenate,
    CloudpickleWrapper,
    clear_mpi_env_vars,
)


class AsyncVectorEnv(AsyncVectorEnvBase):
    def __init__(self, env_fns, **kwargs):
        super().__init__(env_fns, **kwargs)
        
    def __init__(
        self,
        env_fns,
        dummy_env_fn=None,
        observation_space=None,
        action_space=None,
        shared_memory=True,
        copy=True,
        context=None,
        daemon=True,
        worker=None,
    ):
        
        ctx = mp.get_context(context)
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy

        # Added dummy_env_fn to fix OpenGL error in Mujoco
        # disable any OpenGL rendering in dummy_env_fn, since it
        # will conflict with OpenGL context in the forked child process
        if dummy_env_fn is None:
            dummy_env_fn = env_fns[0]
        dummy_env = dummy_env_fn()
        self.metadata = dummy_env.metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or dummy_env.observation_space
            action_space = action_space or dummy_env.action_space
        dummy_env.close()
        del dummy_env
        
        super().__init__(env_fns, shared_memory=shared_memory, copy=copy, context=context, daemon=daemon, worker=worker)
        # print(f"self.single_observation_space: {self.single_observation_space}")
        # print(f"self.num_envs: {self.num_envs}")
        # print(f"self.single_observaiton_spacetype: {type(self.single_observation_space)}")
        # input()
        
        # if self.shared_memory:
        #     try:
        #         _obs_buffer = create_shared_memory(
        #             self.single_observation_space, n=self.num_envs, ctx=ctx
        #         )
        #         self.observations = read_from_shared_memory(
        #             _obs_buffer, self.single_observation_space, n=self.num_envs
        #         )
        #     except CustomSpaceError:
        #         raise ValueError(
        #             "Using `shared_memory=True` in `AsyncVectorEnv` "
        #             "is incompatible with non-standard Gym observation spaces "
        #             "(i.e. custom spaces inheriting from `gym.Space`), and is "
        #             "only compatible with default Gym spaces (e.g. `Box`, "
        #             "`Tuple`, `Dict`) for batching. Set `shared_memory=False` "
        #             "if you use custom observation spaces."
        #         )
        # else:
        #     _obs_buffer = None
        #     self.observations = create_empty_array(
        #         self.single_observation_space, n=self.num_envs, fn=np.zeros
        #     )

        # self.parent_pipes, self.processes = [], []
        # self.error_queue = ctx.Queue()
        # target = _worker_shared_memory if self.shared_memory else _worker
        # target = worker or target
        # with clear_mpi_env_vars():
        #     for idx, env_fn in enumerate(self.env_fns):
        #         parent_pipe, child_pipe = ctx.Pipe()
        #         process = ctx.Process(
        #             target=target,
        #             name="Worker<{0}>-{1}".format(type(self).__name__, idx),
        #             args=(
        #                 idx,
        #                 CloudpickleWrapper(env_fn),
        #                 child_pipe,
        #                 parent_pipe,
        #                 _obs_buffer,
        #                 self.error_queue,
        #             ),
        #         )

        #         self.parent_pipes.append(parent_pipe)
        #         self.processes.append(process)

        #         process.daemon = daemon
        #         process.start()
        #         child_pipe.close()

        # self._state = AsyncState.DEFAULT
        # self._check_observation_spaces()
                
    def call_each(self, name: str, 
            args_list: list=None, 
            kwargs_list: list=None, 
            timeout = None):
        n_envs = len(self.parent_pipes)
        if args_list is None:
            args_list = [[]] * n_envs
        assert len(args_list) == n_envs

        if kwargs_list is None:
            kwargs_list = [dict()] * n_envs
        assert len(kwargs_list) == n_envs

        # send
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `call_async` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for i, pipe in enumerate(self.parent_pipes):
            pipe.send(("_call", (name, args_list[i], kwargs_list[i])))
        self._state = AsyncState.WAITING_CALL

        # receive
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `call_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results

    def _poll(self, timeout=None):
        self._assert_is_running()
        if timeout is None:
            return True
        end_time = time.perf_counter() + timeout
        delta = None
        for pipe in self.parent_pipes:
            delta = max(end_time - time.perf_counter(), 0)
            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

def _worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation = env.reset()
                pipe.send((observation, True))
            elif command == "step":
                observation, reward, done, info = env.step(data)
                # if done:
                #     observation = env.reset()
                pipe.send(((observation, reward, done, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))

            elif command == "_check_observation_space":
                pipe.send((data == env.observation_space, True))
            else:
                raise RuntimeError(
                    "Received unknown command `{0}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, "
                    "`_check_observation_space`}.".format(command)
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation = env.reset()
                write_to_shared_memory(
                    index, observation, shared_memory, observation_space
                )
                pipe.send((None, True))
            elif command == "step":
                observation, reward, done, info = env.step(data)
                # if done:
                #     observation = env.reset()
                write_to_shared_memory(
                    index, observation, shared_memory, observation_space
                )
                pipe.send(((None, reward, done, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_observation_space":
                pipe.send((data == observation_space, True))
            else:
                raise RuntimeError(
                    "Received unknown command `{0}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, "
                    "`_check_observation_space`}.".format(command)
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()