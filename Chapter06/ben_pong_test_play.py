import gym
import time
import argparse
import numpy as np
import collections
from lib import wrappers
from lib import wrappers_ben as wb


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25

if __name__ == "__main__":
    env = wb.make_env(DEFAULT_ENV_NAME)
    env = gym.wrappers.Monitor(env,"recording", force=True)
    env.reset()
    while True:
        start_ts = time.time()
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        if done:
            break
#         delta = 1/FPS - (time.time() - start_ts)
#         if delta > 0:
#             time.sleep(delta)
    env.env.close()