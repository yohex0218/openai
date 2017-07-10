# -*- coding: utf-8 -*-
import sys
import random
import numpy
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import gym


class DQNAgent:
    def __init__(self, state_count, action_count, memory_cap, batch_size):
        self.state_count = state_count
        self.action_count = action_count
        self.model = self._create_model()
        self.samples = deque(maxlen=memory_cap)
        self.batch_size = batch_size
        self.gamma = 0.99
        self.epsilon = 1.
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.is_learning = True

    def _create_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_count, activation='relu'))
        model.add(Dense(self.action_count, activation='linear'))
        opt = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=opt)
        return model

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_count - 1)
        else:
            q = self.model.predict(state.reshape(1, self.state_count)).flatten()
            return numpy.argmax(q)

    def observe(self, sample):
        self.samples.append(sample)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        if not self.is_learning:
            return

        batch_length, batch = self._sample()
        no_state = numpy.zeros(self.state_count)
        s = numpy.array([o[0] for o in batch])
        s_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])
        q = self.model.predict(s)
        q_ = self.model.predict(s_)
        x = numpy.zeros((batch_length, self.state_count))
        y = numpy.zeros((batch_length, self.action_count))

        for i in range(batch_length):
            observation = batch[i]
            s = observation[0]
            a = observation[1]
            r = observation[2]
            s_ = observation[3]

            t = q[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + self.gamma * numpy.amax(q_[i])

            x[i] = s
            y[i] = t

        self.model.fit(x, y, batch_size=self.batch_size, epochs=1, verbose=0)

    def _sample(self):
        n = min(self.batch_size, len(self.samples))
        return n, random.sample(self.samples, n)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class Environment:
    def __init__(self, problem):
        self.env = gym.make(problem)
        self.goal = self.env.spec.timestep_limit
        self.results = deque(maxlen=5)
        self.count = 0

    def run(self, agent):
        s = self.env.reset()
        total_r = 0
        while True:
            self.env.render()
            a = agent.act(s)
            s_, r, done, info = self.env.step(a)
            if done:
                s_ = None
            agent.observe((s, a, r, s_))
            agent.replay()

            s = s_
            total_r += r
            if done:
                self.count += 1
                self.results.append(self.goal == total_r)
                if agent.is_learning and numpy.all(self.results):
                    print("Stop Learning.")
                    agent.is_learning = False
                    agent.save("./save/{}-BEST.h5".format(env_name))
                break

        print("Game:{} Reward:{}".format(self.count, total_r))


env_name = 'CartPole-v0'
# env_name = 'CartPole-v1'

env = Environment(env_name)
state_count = env.env.observation_space.shape[0]
action_count = env.env.action_space.n

agent = DQNAgent(state_count, action_count, memory_cap=10000, batch_size=64)
# agent.load("./save/{}-BEST.h5".format(env_name))
# agent.is_learning = False
# agent.epsilon = 0.

try:
    while True:
        env.run(agent)
except KeyboardInterrupt:
    sys.exit(0)
