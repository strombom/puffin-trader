
import numpy as np

from BitmexSim.bitmex_env import BitmexEnv


if __name__ == '__main__':
    bitmex_env = BitmexEnv(max_steps=100)

    bitmex_env.reset()

    observation, reward, done = bitmex_env.step(None)

    observation = np.array([[observation]])

    print("observation", observation.shape)

    bitmex_env.render()

    bitmex_env.close()

    print('a')

