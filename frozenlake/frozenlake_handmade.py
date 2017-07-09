import gym
from gym import wrappers

# 4x4 MAP
# SFFF
# FHFH
# FFFH
# HFFG

# 8x8 MAP
# SFFFFFFF
# FFFFFFFF
# FFFHFFFF
# FFFFFHFF
# FFFHFFFF
# FHHFFFHF
# FHFFHFHF
# FFFHFFFG


# 0:L, 1:D, 2:R, 3:U
ACTIONS_MAP = {
    "FrozenLake-v0": (
        0, 3, 3, 3,
        0, -1, 0, -1,
        3, 1, 0, -1,
        -1, 2, 1, -1,
    ),
    "FrozenLake8x8-v0": (
        2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 2,
        -1, -1, -1, -1, -1, -1, -1, 2,
        -1, -1, -1, -1, -1, -1, -1, 2,
        -1, -1, -1, -1, -1, -1, -1, 2,
        -1, -1, -1, -1, -1, -1, -1, 2,
        -1, -1, -1, -1, -1, -1, -1, 2,
        -1, -1, -1, -1, -1, -1, -1, -1,
    )
}


def get_action(actions, ob):
    act = actions[ob]
    assert 0 <= act <= 3
    return act


def main():
    # env_name = "FrozenLake-v0"
    env_name = "FrozenLake8x8-v0"

    actions = ACTIONS_MAP[env_name]
    rec_dir = None

    test_count = 100000

    env = gym.make(env_name)
    if rec_dir:
        wrappers.Monitor(env, rec_dir)

    reward_total = 0.0
    for episode in range(test_count):
        ob = env.reset()
        while True:
            ob, reward, done, info = env.step(get_action(actions, ob))
            if done:
                reward_total += reward
                break

    print("episodes:       {}".format(test_count))
    print("total reward:   {}".format(reward_total))
    print("average reward: {:.2f}".format(reward_total / test_count))


if __name__ == "__main__":
    main()
