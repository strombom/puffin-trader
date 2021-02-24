
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from enum import IntEnum

from BinanceSim.binance_simulator import BinanceSimulator
from Common.Misc import PositionDirection
from position import Position
from plotter import Plotter
from slopes import Slopes


if __name__ == '__main__':
    with open(f"cache/intrinsic_time_runner.pickle", 'rb') as f:
        delta, runner = pickle.load(f)

    runner.ie_prices = np.array(runner.ie_prices)[0:10000]
    x = np.arange(runner.ie_prices.shape[0])
    slopes = Slopes(runner.ie_prices, use_cache=True)

    slopes_history_count = 0
    first_idx = Slopes.max_slope_length + slopes_history_count
    simulator = BinanceSimulator(initial_usdt=0.0, initial_btc=1.0, max_leverage=2, mark_price=runner.ie_prices[0], initial_leverage=0.0)
    plotter = Plotter(slopes=slopes)
    position = Position(delta=delta, direction=PositionDirection.long, plotter=plotter)

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
