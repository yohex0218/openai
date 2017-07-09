import sys
import tty
import termios
import gym


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# LEFT:0, DOWN:1, RIGHT:2, UP:3
action_keys = {"\x1b[A": UP, "\x1b[B": DOWN, "\x1b[C": RIGHT, "\x1b[D": LEFT}


class KeyboardMonitor:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

input_key = KeyboardMonitor()


def get_action():
    key = input_key()
    if key not in action_keys.keys():
        print("===== G A M E O V E R =====")
        sys.exit(0)

    return action_keys[key]


def main():
    env_name = "FrozenLake-v0"
    # env_name = "FrozenLake8x8-v0"

    env = gym.make(env_name)

    print("observation space :", env.observation_space)
    print("action space :", env.action_space)
    print("timestep limit :", env.spec.timestep_limit)

    while True:
        env.reset()
        env.render()
        while True:
            ob, reward, done, info = env.step(get_action())
            env.render()
            if done:
                if reward > 0:
                    print("===== G O O D J O B ! =====")
                else:
                    print("===== G A M E O V E R =====")
                break
            else:
                print("Observation: {} / Reward : {} / Info : {}".format(ob, reward, info))


if __name__ == "__main__":
    main()

