import random
import os.path

import numpy as np
import gym
from gym import wrappers


class Agent:
    def __init__(self, env, epsilon):
        self.env = env
        self.epsilon = epsilon
        self.q = np.zeros([env.observation_space.n, env.action_space.n])

    def train(self, alpha, gamma):
        """1エピソード学習"""
        state = self.env.reset()

        while True:
            # 現在のQ関数に基づくe-greedy
            act = self._e_greedy(state)
            state_next, reward, done, info = self.env.step(act)

            # qは0で初期化しているので、state_nextが終端状態なら
            # q_next_maxは0になる
            q_next_max = np.max(self.q[state_next])
            self.q[state][act] = (1 - alpha) * self.q[state][act] + alpha * (reward + gamma * q_next_max)

            if done:
                return reward
            else:
                state = state_next

    def _e_greedy(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # return np.argmax(self.q[state])
            # 同点の場合は同点のものからランダムに選ぶ
            max_q = np.max(self.q[state])
            return random.choice([idx for idx, value in enumerate(self.q[state]) if value == max_q])

    def evaluate(self):
        """学習結果を用いて1エピソード実行"""
        state = self.env.reset()

        while True:
            act = np.argmax(self.q[state])
            state, reward, done, info = self.env.step(act)

            if done:
                return reward


def main():
    env_name = "FrozenLake-v0"
    # env_name = "FrozenLake8x8-v0"

    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1

    train_count = 10000
    test_count = 1000
    rec_dir = None

    env = gym.make(env_name)
    print("# step-max: {}".format(env.spec.timestep_limit))

    if rec_dir:
        subdir = "{}-alpha{}-gamma{}-eps{}-learn{}-test{}".format(
            env_name, alpha, gamma, epsilon, train_count, test_count
        )
        wrappers.Monitor(env, os.path.join(rec_dir, subdir))

    agent = Agent(env, epsilon)

    print("##### LEARNING #####")
    reward_total = 0.0
    for episode in range(train_count):
        reward_total += agent.train(alpha, gamma)

    print("Q TABLE : LEFT, DOWN, RIGHT, UP")
    print(agent.q)

    print("episodes:       {}".format(train_count))
    print("total reward:   {}".format(reward_total))
    print("winning ratio:  {:.2f}".format(reward_total / train_count))

    print("##### TEST #####")
    reward_total = 0.0
    for episode in range(test_count):
        reward_total += agent.evaluate()
    print("episodes:       {}".format(test_count))
    print("total reward:   {}".format(reward_total))
    print("winning ratio:  {:.2f}".format(reward_total / test_count))


if __name__ == "__main__":
    main()
