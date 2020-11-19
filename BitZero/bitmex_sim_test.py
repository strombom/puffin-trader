
import numpy as np

from BitmexSim.bitmex_sim import BitmexSim


if __name__ == '__main__':
    bitsim = BitmexSim(max_steps=100)

    bitsim.reset()

    observation, reward, done = bitsim.step(None)

    observation = np.array([[observation]])

    bitsim.render()

    bitsim.close()

    print('a')

