import pickle
import numpy as np
import gym
from gym import spaces

from bybit_simulator import ByBitSimulator


class TradeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    symbol = "BTCUSDT"
    simulator = None
    indicator_idx = 0
    step_count = 0

    def __init__(self):
        super(TradeEnv, self).__init__()    # Define action and observation space
        # They must be gym.spaces objects    # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)  # 0: sell, 1: hodl, 2: buy
        self.observation_space = spaces.Box(low=-10, high=10, shape=(62,), dtype=np.float)

        with open("indicators.pickle", 'rb') as f:
            training_data = pickle.load(f)
            self.indicators = training_data['indicators']
            self.intrinsic_events = training_data['intrinsic_events']

    def _get_observation(self):
        relative_price = 0.0
        position_age = 0.0
        position_buy_size = 0.0
        position_sell_size = 0.0

        position_buy = self.simulator.positions_buy[self.symbol]
        position_sell = self.simulator.positions_sell[self.symbol]
        if position_buy.size > 0:
            relative_price = (self.simulator.mark_price[self.symbol] - position_buy.entry_price) / position_buy.entry_price
            relative_age = np.tanh(position_buy.age())
        elif position_sell.size < 0:
            relative_price = (self.simulator.mark_price[self.symbol] - position_sell.entry_price) / position_sell.entry_price
            relative_age = np.tanh(position_buy.age())

        self.indicators[self.indicator_idx][:4] = [relative_price, position_age, position_buy_size, position_sell_size]
        return self.indicators[self.indicator_idx]

    def step(self, action):
        # ['timestamp', 'price', 'size', 'tick_id']
        self.simulator.mark_price[self.symbol] = self.intrinsic_events['price'].iloc[self.indicator_idx]

        if action == 0:  # sell
            qty = self.simulator.calculate_order_size(leverage=-1, symbol=self.symbol)
            self.simulator.market_order(timestamp=self.indicator_idx, order_size=qty, symbol=self.symbol)
        elif action == 1:  # hodl
            pass
        elif action == 2:  # buy
            qty = self.simulator.calculate_order_size(leverage=1, symbol=self.symbol)
            self.simulator.market_order(timestamp=self.indicator_idx, order_size=qty, symbol=self.symbol)

        _observation = self._get_observation()
        self.indicator_idx += 1
        self.step_count += 1

        _reward = self.step_count
        _done = False
        if self.step_count == 200:
            _done = True
        return _observation, _reward, _done, {}

    def reset(self):
        self.simulator = ByBitSimulator(initial_usdt=10000, symbols=[self.symbol])
        self.indicator_idx = self.intrinsic_events.shape[0] - self.indicators.shape[0]
        self.step_count = 0
        return self._get_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass


if __name__ == '__main__':
    env = TradeEnv()
    obs = env.reset()
    print(obs)
    action = 1
    obs, reward, done, _ = env.step(action)
    print(obs, reward, done)
