
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from enum import IntEnum

from BinanceSim.binance_simulator import BinanceSimulator
from Common.Misc import PositionDirection
from position import Position
from plotter import Plotter
from slopes import Slopes


if __name__ == '__main__':
    # with open(f"cache/intrinsic_time_runner.pickle", 'rb') as f:
    #     delta, runner = pickle.load(f)

    delta = 0.004
    runner_data = pd.read_csv('../tmp/binance_runner.csv')

    maxrow = np.argmax(runner_data['price'].to_numpy())

    print(maxrow)
    print(runner_data.loc[maxrow]['price'])
    print(runner_data.loc[maxrow])
    quit()

    runner_prices = runner_data['price'].to_numpy()
    runner_durations = runner_data['duration'].to_numpy()

    x = np.arange(runner_data.shape[0])
    slopes = Slopes(runner_prices, use_cache=False)

    slopes_history_count = 0
    first_idx = Slopes.max_slope_length + slopes_history_count
    simulator = BinanceSimulator(initial_usdt=0.0, initial_btc=1.0, max_leverage=2, mark_price=runner_prices[0], initial_leverage=0.0)
    plotter = Plotter(slopes=slopes)
    position = Position(delta=delta, direction=PositionDirection.long, plotter=plotter)

    for idx, ie_price in enumerate(runner_prices[:first_idx]):
        plotter.append_event(PositionDirection.hedge, idx, ie_price)
        plotter.append_timestamp(idx, runner_data.iloc[idx]['timestamp'])

    previous_trade_value = simulator.get_value_usdt(mark_price=runner_prices[first_idx])

    previous_make_trade = False

    for idx in range(first_idx, runner_prices.shape[0]):
        slope = slopes.slopes.iloc[idx - Slopes.max_slope_length]
        ie_price = runner_prices[idx]

        plotter.append_slope_length(idx, slope['length'])
        plotter.append_volatility(idx, slope['volatility'])

        previous_duration = runner_durations[idx - 1]
        duration = runner_durations[idx]

        make_trade = position.step(idx, ie_price, runner_prices[idx - 1], duration, previous_duration, slope)

        if make_trade and previous_make_trade:
            if position.direction == PositionDirection.short:
                order_size = simulator.calculate_order_size_btc(leverage=2.5, mark_price=ie_price)
                simulator.market_order(order_size_btc=order_size, mark_price=ie_price)
            else:
                order_size = simulator.calculate_order_size_btc(leverage=-1.5, mark_price=ie_price)
                simulator.market_order(order_size_btc=order_size, mark_price=ie_price)
            position.direction *= -1
            made_trade = True

            value_after = simulator.get_value_usdt(mark_price=ie_price)
            profit = (value_after - previous_trade_value) / previous_trade_value
            previous_trade_value = value_after

            plotter.append_annotation(x=idx, y=ie_price, direction=PositionDirection.short, profit=profit)

        previous_make_trade = make_trade

        plotter.append_value(idx, simulator.get_value_usdt(mark_price=ie_price))
        plotter.append_angle(idx, slope.angle)
        plotter.append_event(position.direction, idx, ie_price)
        plotter.append_timestamp(idx, runner_data.iloc[idx]['timestamp'])

    plotter.plot()
