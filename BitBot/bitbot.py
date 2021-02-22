
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from enum import IntEnum

from BinanceSim.binance_simulator import BinanceSimulator
from Common.Misc import PositionDirection, Regime
from plotter import Plotter
from slopes import Slopes, Slope


class Position:
    def __init__(self, direction: PositionDirection, plotter: Plotter):
        self.direction = direction
        self.plotter = plotter
        self.regime = Regime.chop

    def step(self, ie_idx: int, mark_price: float, mark_price_prev: float, slope: Slope) -> bool:
        # regime
        if self.regime == Regime.chop:
            if (slope.angle > 0.25 and mark_price > mark_price_prev) or \
                    (slope.angle < -0.25 and mark_price < mark_price_prev):
                if slope.length > 0.20 and slope.volatility > 0.3 - 0.2 * slope.length:
                    self.regime = Regime.trend
                    self.plotter.regime_change(x=ie_idx, mark_price=mark_price, regime=self.regime)

        # threshold_delta = (1.6 + 0.8 * (slope_len - 10) / 70) * delta
        threshold_delta = 1.85 * delta

        if abs(slope.angle) > 2 and slope.length > 0.4:
            threshold_delta *= 0.9
        # elif abs(slope_angle) > 1:
        #     threshold_delta *= 1.1

        # threshold_delta *= (1 - max(0.5, min(1, abs(anglediff))) / 2)

        make_trade = False

        angle_threshold = 0.2 / 5
        prev_slope_angle = 0
        if slopes[-2] is not None:
            prev_slope_angle = slopes[ie_idx - Slopes.max_slope_length - 1].angle

        if position.direction == PositionDirection.short:
            threshold = slope.y[-1] * (1 + threshold_delta)
            self.plotter.append_threshold(ie_idx, threshold)

            if mark_price > threshold or \
                    (mark_price > slope.y[-1] and slope.angle > angle_threshold and slope.angle > prev_slope_angle and mark_price > mark_price_prev) or \
                    (mark_price > slope.y[-1] and slope.angle > 0 and slope.length > 0.3 and mark_price > mark_price_prev):
                make_trade = True

        elif position.direction == PositionDirection.long:
            threshold = slope.y[-1] * (1 - threshold_delta)
            self.plotter.append_threshold(ie_idx, threshold)

            if mark_price < threshold or \
                    (mark_price < slope.y[-1] and slope.angle < -angle_threshold and slope.angle < prev_slope_angle and mark_price < mark_price_prev) or \
                    (mark_price < slope.y[-1] and slope.angle < 0 and slope.length > 0.3 and mark_price < mark_price_prev):
                make_trade = True

        # if abs(anglediff) > 1.0:
        #     threshold_delta *= 1 - 0.5
        # elif abs(anglediff) > 0.7:
        #     threshold_delta *= 1 - 0.3
        # elif abs(anglediff) > 0.4:
        #     threshold_delta *= 1 - 0.2

        if make_trade and self.regime == Regime.trend:
            self.regime = Regime.chop
            self.plotter.regime_change(x=ie_idx, mark_price=mark_price, regime=self.regime)

        return make_trade


if __name__ == '__main__':
    with open(f"cache/intrinsic_time_runner.pickle", 'rb') as f:
        delta, runner = pickle.load(f)

    runner.ie_prices = np.array(runner.ie_prices)[0:10000]
    x = np.arange(runner.ie_prices.shape[0])
    slopes = Slopes(runner.ie_prices, use_cache=False)

    slopes_history_count = 0
    first_idx = Slopes.max_slope_length + slopes_history_count
    simulator = BinanceSimulator(initial_usdt=0.0, initial_btc=1.0, max_leverage=2, mark_price=runner.ie_prices[0], initial_leverage=0.0)
    plotter = Plotter(slopes)
    position = Position(direction=PositionDirection.long, plotter=plotter)

    for idx, ie_price in enumerate(runner.ie_prices[:first_idx]):
        plotter.append_event(PositionDirection.hedge, idx, ie_price)

    previous_trade_value = simulator.get_value_usdt(mark_price=runner.ie_prices[first_idx])

    for idx in range(first_idx, runner.ie_prices.shape[0]):
        slope = slopes[idx - Slopes.max_slope_length]
        ie_price = runner.ie_prices[idx]

        plotter.append_slope_length(idx, slope.length)
        plotter.append_volatility(idx, slope.volatility)

        make_trade = position.step(idx, ie_price, runner.ie_prices[idx - 1], slope)  # , slopes[idx-slopes_history_count:idx])

        if make_trade:
            if position.direction == PositionDirection.short:
                order_size = simulator.calculate_order_size_btc(leverage=1.0, mark_price=ie_price)
                simulator.market_order(order_size_btc=order_size, mark_price=ie_price)
            else:
                order_size = simulator.calculate_order_size_btc(leverage=-1.0, mark_price=ie_price)
                simulator.market_order(order_size_btc=order_size, mark_price=ie_price)
            position.direction *= -1
            made_trade = True

            value_after = simulator.get_value_usdt(mark_price=ie_price)
            profit = (value_after - previous_trade_value) / previous_trade_value
            previous_trade_value = value_after

            plotter.append_annotation(x=idx, y=ie_price, direction=PositionDirection.short, profit=profit)

        plotter.append_value(idx, simulator.get_value_usdt(mark_price=ie_price))
        plotter.append_angle(idx, slope.angle)
        plotter.append_event(position.direction, idx, ie_price)

    plotter.plot()
