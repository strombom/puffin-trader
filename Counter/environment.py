import pickle
import numpy as np
import gym
from gym import spaces
from gym.envs.classic_control import rendering

from bybit_simulator import ByBitSimulator


class PlotLog:
    def __init__(self):
        self.prices = []


class TradeEnv(gym.Env):
    metadata = {'render.modes': ['human', "rgb_array"]}
    symbol = "BTCUSDT"

    def __init__(self):
        super(TradeEnv, self).__init__()

        self.action_space = spaces.Discrete(3)  # 0: sell, 1: hodl, 2: buy
        self.observation_space = spaces.Box(low=-10, high=10, shape=(60,), dtype=np.float)

        self.simulator = None
        self.indicator_idx = 0
        self.step_count = 0
        self.max_step_count = 100
        self.plot_log = PlotLog()

        self.viewer = None
        self.state = None
        self.carttrans = None

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
            position_age = np.tanh(position_buy.get_age(self.indicator_idx))
            position_buy_size = 1.0
        elif position_sell.size < 0:
            relative_price = (self.simulator.mark_price[self.symbol] - position_sell.entry_price) / position_sell.entry_price
            position_age = np.tanh(position_buy.get_age(self.indicator_idx))
            position_sell_size = 1.0

        self.state = [0.1, 0.2, 0.3, 0.4]

        self.indicators[self.indicator_idx][:4] = [relative_price, position_age, position_buy_size, position_sell_size]
        return self.indicators[self.indicator_idx]

    def step(self, action):
        # ['timestamp', 'price', 'size', 'tick_id']
        mark_price = self.intrinsic_events['price'].iloc[self.indicator_idx]
        self.simulator.mark_price[self.symbol] = mark_price

        if action == 0 and self.simulator.positions_sell[self.symbol].size == 0:  # sell
            qty = self.simulator.calculate_order_size(leverage=-1, symbol=self.symbol)
            self.simulator.market_order(timestamp=self.indicator_idx, order_size=qty, symbol=self.symbol)
        elif action == 1:  # hodl
            pass
        elif action == 2 and self.simulator.positions_buy[self.symbol].size == 0:  # buy
            qty = self.simulator.calculate_order_size(leverage=1, symbol=self.symbol)
            self.simulator.market_order(timestamp=self.indicator_idx, order_size=qty, symbol=self.symbol)

        _observation = self._get_observation()
        self.indicator_idx += 1
        self.step_count += 1

        equity = self.simulator.get_equity_usdt()

        _reward = equity  # self.step_count + equity
        _done = False
        if equity < 1000.0 or self.step_count == self.max_step_count:
            _done = True

        self.plot_log.prices.append(mark_price)

        return _observation, _reward, _done, {}

    def reset(self):
        self.simulator = ByBitSimulator(initial_usdt=10000, symbols=[self.symbol])
        self.indicator_idx = self.intrinsic_events.shape[0] - self.indicators.shape[0]
        self.step_count = 0
        self.plot_log = PlotLog()
        return self._get_observation()

    def render(self, mode='human', close=False):
        screen_width = 600
        screen_height = 200

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

        if len(self.plot_log.prices) > 1:
            polyline = []
            y_min, y_max = min(self.plot_log.prices), max(self.plot_log.prices)
            y_span = y_max - y_min
            for idx, price in enumerate(self.plot_log.prices):
                x = idx / self.max_step_count * (screen_width - 20) + 10
                y = screen_height - (0 + 180 * (price - y_min) / y_span)
                polyline.append((x, y))
            self.viewer.draw_polyline(polyline, color=(0.5, 0.5, 0.5))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

        """
        self.x_threshold = 1
        self.length = 0.5

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
        """

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = TradeEnv()
    obs = env.reset()
    print(obs)
    action = 1
    obs, reward, done, _ = env.step(action)
    print(obs, reward, done)
