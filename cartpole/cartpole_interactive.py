import sys
import tty
import termios
import gym

LEFT = 0
RIGHT = 1

# RIGHT ARROW:\x1b[C, LEFT ARROW:\x1b[D
action_keys = {"\x1b[C": RIGHT, "\x1b[D": LEFT}


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
    env_name = "CartPole-v0"

    env = gym.make(env_name)
    timestep_limit = env.spec.timestep_limit

    print("observation space :", env.observation_space)
    print("action space :", env.action_space)
    print("timestep limit :", timestep_limit)

    while True:
        s = env.reset()
        count = 0
        while True:
            env.render()
            ob, reward, done, _ = env.step(get_action())
            count += 1
            if done:
                if count == timestep_limit:
                    print("===== G O O D J O B ! =====")
                else:
                    print("===== G A M E O V E R =====")
                break
            else:
                print("Count: {} / Status: {} / Reward : {}".format(count, ob, reward))

if __name__ == "__main__":
    main()
