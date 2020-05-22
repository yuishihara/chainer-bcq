import gym
from gym.wrappers.monitor import Monitor

import numpy as np


class NumpyFloat32Env(gym.Wrapper):
    def __init__(self, env):
        super(NumpyFloat32Env, self).__init__(env)

    def step(self, action):
        next_state, reward, done, metadata = self.env.step(action)
        return np.float32(next_state), np.float32(reward), done, metadata

    def reset(self, **kwargs):
        initial_state = self.env.reset(**kwargs)
        return np.float32(initial_state)


class ScreenRenderEnv(gym.Wrapper):
    def __init__(self, env):
        super(ScreenRenderEnv, self).__init__(env)

    def step(self, action):
        self.env.render()
        return self.env.step(action)

    def reset(self):
        state = self.env.reset()
        self.env.render()
        return state


class EveryEpisodeMonitor(Monitor):
    def __init__(self, env, directory, force=False, resume=False,
                 write_upon_reset=False, uid=None, mode=None):
        super(EveryEpisodeMonitor, self).__init__(env=env, directory=directory,
                                                  video_callable=(
                                                      lambda _: True),
                                                  force=force, resume=resume,
                                                  write_upon_reset=write_upon_reset, uid=uid, mode=mode)
