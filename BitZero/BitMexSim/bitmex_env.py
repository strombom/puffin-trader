import pickle
import random
import numpy as np
from enum import IntEnum
from datetime import datetime

from BitmexSim.bitmex_simulator import BitmexSimulator


class BitmexActions(IntEnum):
    nop = 0
    market_buy = 1
    market_sell = 2
    limit_buy_near = 3
    limit_sell_near = 4
    limit_buy_delta = 5
    limit_sell_delta = 6


class BitmexEnv:
    def __init__(self, max_steps):
        with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
            deltas, order_books, runners, runner_clock, clock_TMV, clock_R = pickle.load(f)

        self.deltas = deltas
        self.runners = runners
        self.runner_clock = runner_clock
        self.clock_TMV = clock_TMV
        self.clock_R = clock_R

        self.max_steps = max_steps
        self.step_count = 0
        self.step_idx = 0
        self.start_price = 0.0

        self.max_leverage = 10.0
        self.simulator = None
        self.action = BitmexActions.nop
        self.action_price = 0.0

        self.previous_value = 0.0

        self.log = {}
        self.reset()

    def seed(self, seed):
        pass

    def step(self, action):
        #print('deltas', len(self.deltas))
        #print('runner_clock.ie_prices', len(self.runner_clock.ie_prices))
        #print('runners[0].ie_prices', len(self.runners[0].ie_prices))
        #print('runners[-1].ie_prices', len(self.runners[-1].ie_prices))
        #print('clock_TMV', self.clock_TMV.shape)
        #print('clock_R', self.clock_R.shape)

        mark_price = self.runner_clock.ie_prices[self.step_idx]

        price_max = self.runner_clock.ie_prices_max[self.step_idx]
        price_min = self.runner_clock.ie_prices_min[self.step_idx]

        # Evaluate limit order
        order_leverage = 0.0
        if self.action == BitmexActions.limit_buy_near or self.action == BitmexActions.limit_buy_delta:
            if price_min < self.action_price:
                order_leverage = self.max_leverage

        if self.action == BitmexActions.limit_sell_near or self.action == BitmexActions.limit_sell_near:
            if price_max > self.action_price:
                order_leverage = -self.max_leverage

        if order_leverage != 0.0:
            order_size = self.simulator.calculate_order_size(leverage=order_leverage, mark_price=self.action_price)
            self.simulator.market_order(order_contracts=order_size, mark_price=self.action_price)

        # Evaluate market order
        order_leverage = 0.0
        if action == BitmexActions.market_buy:
            order_leverage = self.max_leverage
        elif action == BitmexActions.market_sell:
            order_leverage = -self.max_leverage

        if order_leverage != 0.0:
            order_size = self.simulator.calculate_order_size(leverage=order_leverage, mark_price=mark_price)
            self.simulator.market_order(order_contracts=order_size, mark_price=mark_price)

        value = self.simulator.get_value(mark_price=mark_price)
        reward = (value - self.previous_value) / self.previous_value * 100
        self.previous_value = value

        self.step_idx += 1
        done = self.step_idx == len(self.runner_clock.ie_prices)
        if value < self.start_price * 0.1:
            done = True

        leverage = self.simulator.get_leverage(mark_price=mark_price)
        observation = np.concatenate((self.clock_TMV[:, self.step_idx], self.clock_R[:, self.step_idx], [leverage]))

        self.action = action
        if self.action == BitmexActions.limit_buy_near:
            self.action_price = mark_price - 0.5
        elif self.action == BitmexActions.limit_buy_delta:
            self.action_price = mark_price * (1 - self.deltas[0]) + 0.5
        elif self.action == BitmexActions.limit_sell_near:
            self.action_price = mark_price + 0.5
        elif self.action == BitmexActions.limit_sell_near:
            self.action_price = mark_price * (1 + self.deltas[0]) - 0.5
        else:
            self.action_price = 0.0

        self.log['timestamp'].append(self.runner_clock.ie_times[self.step_idx])
        self.log['price'].append(mark_price)
        self.log['leverage'].append(self.simulator.get_leverage(mark_price=mark_price))
        self.log['value'].append(new_value)
        self.log['action'].append(self.action)
        self.log['action_price'].append(self.action_price)

        return observation, reward, done, None

    def reset(self):
        self.step_count = 0
        self.action = BitmexActions.nop
        self.action_price = 0.0

        filepath = f'log/log_{datetime.now().strftime("%Y-%d-%m_%H%M%S")}.csv'
        with open(filepath, 'w') as f:
            for idx in range(len(self.log['timestamp'])):
                f.write(f'{self.log["timestamp"][idx]},')
                f.write(f'{self.log["price"][idx]},')
                f.write(f'{self.log["leverage"][idx]},')
                f.write(f'{self.log["value"][idx]},')
                f.write(f'{self.log["action"][idx]},')
                f.write(f'{self.log["action_price"][idx]}\n')

        self.log = {'timestamp': [],
                    'price': [],
                    'leverage': [],
                    'value': [],
                    'action': [],
                    'action_price': []}

        self.step_idx = random.randint(0, len(self.runner_clock.ie_prices) - self.max_steps - 1)
        self.start_price = self.runner_clock.ie_prices[self.step_idx]
        self.simulator = BitmexSimulator(max_leverage=self.max_leverage, mark_price=self.runner_clock.ie_prices[0])
        self.previous_value = self.simulator.get_value(mark_price=self.start_price)

        return self.step(0)[0]

    def render(self):
        pass

    def close(self):
        pass
