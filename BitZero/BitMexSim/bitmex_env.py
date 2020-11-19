import pickle
import random
import numpy as np
from enum import Enum

from BitmexSim.bitmex_simulator import BitmexSimulator


class BitmexActions(Enum):
    nop = 0
    market_buy = 1
    market_sell = 2


class BitmexEnv:
    def __init__(self, max_steps):
        with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
            deltas, order_books, runners, runner_clock, clock_TMV, clock_R = pickle.load(f)

        self.deltas = deltas
        self.order_books = order_books
        self.runners = runners
        self.runner_clock = runner_clock
        self.clock_TMV = clock_TMV
        self.clock_R = clock_R

        self.max_steps = max_steps
        self.step_count = 0
        self.step_idx = 0

        self.max_leverage = 10.0

        self.simulator = BitmexSimulator(max_leverage=self.max_leverage, mark_price=order_books[0].mid)
        self.reset()

    def seed(self, seed):
        pass

    def step(self, action):

        print('deltas', len(self.deltas))
        print('order_books', len(self.order_books))
        print('runner_clock.ie_prices', len(self.runner_clock.ie_prices))
        print('runners[0].ie_prices', len(self.runners[0].ie_prices))
        print('runners[-1].ie_prices', len(self.runners[-1].ie_prices))
        print('clock_TMV', self.clock_TMV.shape)
        print('clock_R', self.clock_R.shape)

        mark_price = self.order_books[self.step_idx].mid
        previous_value = self.simulator.get_value(mark_price=mark_price)

        order_leverage = 0.0
        if action == BitmexActions.market_buy:
            order_leverage = self.max_leverage
        elif action == BitmexActions.market_sell:
            order_leverage = -self.max_leverage

        if order_leverage != 0.0:
            order_size = self.simulator.calculate_order_size(leverage=order_leverage, mark_price=mark_price)
            self.simulator.market_order(order_contracts=order_size, mark_price=mark_price)

        self.step_idx += 1
        mark_price = self.order_books[self.step_idx].mid
        new_value = self.simulator.get_value(mark_price=mark_price)

        reward = new_value - previous_value

        done = self.step_idx == len(self.runner_clock.ie_prices)

        leverage = self.simulator.get_leverage(mark_price=mark_price)
        observation = np.concatenate((self.clock_TMV[:, self.step_idx], self.clock_R[:, self.step_idx], [leverage]))

        print('observation', observation)

        return observation, reward, done

    def reset(self):
        self.step_count = 0
        self.step_idx = random.randint(0, len(self.runner_clock.ie_prices) - self.max_steps - 1)

        return self.step(0)[0]

    def render(self):
        pass

    def close(self):
        pass
