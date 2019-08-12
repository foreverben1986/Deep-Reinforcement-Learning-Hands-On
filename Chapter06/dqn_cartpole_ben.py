from lib import wrappers_ben as wb
from lib import dqn_model_ben_cartpole as dmb

import time
import numpy as np
import collections
import gym
from collections import namedtuple

#Hyparameter
DEFAULT_ENV_NAME = "CartPole-v1"
CAPACITY = 1000
GAMMA = 0.99
BATCH_SIZE = 6
EPSILON_START = 1.0
EPSILON_DECAY_LAST_FRAME = 10**4
EPSILON_FINAL = 0.02
EPISODES = 40000
START_STEPS = 1000
COPY_STEP = 100
REPORT_EVERY_STEP = 1000


Experience = namedtuple('Experience', field_names=['obs', 'action', 'reward', 'done', 'next_obs'])
class Agent:
    def __init__(self, env):
        self.env = env
        self.exp_buffer = collections.deque(maxlen=CAPACITY)
        self._reset()
    def _reset(self):
        self.current_obs = self.env.reset()
        self.total_reward = 0.0
    def get_Qas(self, model, obs):
        Qas = model.predict(np.array([obs]), 1)
        return Qas
    def select_epision_greedy_action(self, model, obs, epsilon):
        Qas = self.get_Qas(model, obs)
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(np.squeeze(Qas))
        return action
    def play_step(self, model, epsilon):
        action = self.select_epision_greedy_action(model, self.current_obs, epsilon)
        new_obs,reward,done,_ = self.env.step(action)
        exp = Experience(self.current_obs, action, reward, done, new_obs)
        self.exp_buffer.append(exp)
        self.current_obs = new_obs
        if done:
            self._reset()
    def test(self, DEFAULT_ENV_NAME, model, i_epoch):
        env = gym.make(DEFAULT_ENV_NAME)
        env = gym.wrappers.Monitor(env,"recording"+str(i_epoch), force=True)
        current_obs = env.reset()
        total_reward = 0
        while True:
            action = self.select_epision_greedy_action(model, current_obs, 0)
            new_obs,reward,done,_ = env.step(action)
            total_reward += reward
            if done:
                break
            current_obs = new_obs
        env.env.close()
        return total_reward
    def get_exp_buffer(self):
        return self.exp_buffer

def sample_memories2(buffer, batch_size):
        indices = np.random.choice(len(buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

def one_hot(target, shape):
    b = np.zeros(shape)
    b[np.arange(shape[0]), target] = 1
    return b

if __name__ == "__main__":
    # train
    env = gym.make(DEFAULT_ENV_NAME)
    myModel = dmb.MyModel(env.observation_space.shape, env.action_space.n)
    agent = Agent(env)
    his_myModel = myModel.export_model()
    for i in range(EPISODES):
        epsilon = max(EPSILON_FINAL, EPSILON_START - i / EPSILON_DECAY_LAST_FRAME)
        agent.play_step(myModel, epsilon)
        exp_buffer = agent.get_exp_buffer()
        if i > START_STEPS:
            memories = sample_memories2(exp_buffer, BATCH_SIZE)
            current_obs_v = memories[0]
            action_v = memories[1]
            reward_v = memories[2]
            done_v = memories[3]
            next_obs_v = memories[4]
            Qas = myModel.predict(current_obs_v, BATCH_SIZE)
            Qas_next = his_myModel.predict(next_obs_v)
            max_reward_v = np.max(Qas_next, axis=-1)
            expect = reward_v + (GAMMA * max_reward_v) * (1 - done_v)
            indicis = one_hot(action_v, Qas.shape)
            Qas[indicis.nonzero()] = expect
            myModel.train(current_obs_v, Qas, BATCH_SIZE, 1)
        if (i + 1) % COPY_STEP == 0 and i > START_STEPS:
            his_myModel.set_weights(myModel.get_weights())
        if (i + 1) % REPORT_EVERY_STEP == 0 and i > START_STEPS:
            total_reward = agent.test(DEFAULT_ENV_NAME, myModel, i)
            print("test total_reward:",total_reward)