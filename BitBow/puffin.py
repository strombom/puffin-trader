import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timezone

from Indicators.supersmoother import SuperSmoother
from rainbow import rainbow_indicator_load_params, rainbow_indicator


class Simulator:
    def __init__(self):
        self.btc = 0.0
        self.usd = 6976.0
        self.fee = 0.00075

    def buy(self, price):
        self.btc = self.usd / price * (1 - self.fee)
        self.usd = 0

    def sell(self, price):
        self.usd = self.btc * price * (1 - self.fee)
        self.btc = 0

    def get_value(self, price):
        return self.usd + self.btc * price


if __name__ == '__main__':
    with open(f"cache/intrinsic_time_runner.pickle", 'rb') as f:
        delta, runner = pickle.load(f)

    # runner.os_prices = np.array(runner.os_prices)
    # runner.dc_prices = np.array(runner.dc_prices)
    runner.ie_prices = np.array(runner.ie_prices)[0:2000]
    x = np.arange(runner.ie_prices.shape[0])

    """
    rainbow_params = rainbow_indicator_load_params()
    rainbow = np.empty(x.shape[0])
    for idx, (timestamp, price) in enumerate(zip(runner.ie_times, runner.ie_prices)):
        timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        rainbow[idx] = rainbow_indicator(params=rainbow_params, timestamp=timestamp, price=price)
    """

    smooths = {}
    """
    smooth_periods = [8]
    for smooth_period in smooth_periods:
        smooth = []
        smoother = SuperSmoother(period=smooth_period, initial_value=runner.ie_prices[0])
        for price in runner.ie_prices:
            smooth.append(smoother.append(price))
        smooths[smooth_period] = smooth
    """


    def estimate_slope(start, stop):
        c_x = np.arange(start, stop)
        r = stats.linregress(c_x, runner.ie_prices[start:stop])
        print('slope', r.slope)
        return c_x, c_x * r.slope + r.intercept

    """
    slope < -10 : 
    """

    simulator = Simulator()
    direction = -1
    state = 'estimate_slope'
    puffin_price = runner.ie_prices[0]
    slope = None
    slope_start_x = 0
    threshold = puffin_price * (1 + 2 * delta)
    thresholds = {'x': [], 'y': []}
    # directions = {'x': [], 'y': []}
    slopes = []
    values = []
    for idx, ie_price in enumerate(runner.ie_prices):
        values.append(simulator.get_value(ie_price))
        if idx < 10:
            thresholds['x'].append(idx)
            thresholds['y'].append(ie_price)
            continue

        slope_start_x = max(slope_start_x, idx - 20)

        if idx - slope_start_x >= 10:
            slope_x, slope_y = estimate_slope(slope_start_x, idx)
            slopes.append({'x': slope_x, 'y': slope_y})

        if direction == -1:
            threshold = slope_y[-1] * (1 + 2 * delta)
            thresholds['x'].append(idx)
            thresholds['y'].append(threshold)

            if ie_price > threshold:
                simulator.buy(ie_price)
                direction *= -1
                # state = 'estimate_slope'
                slope_start_x = idx

        else:
            threshold = slope_y[-1] * (1 - 2 * delta)
            thresholds['x'].append(idx)
            thresholds['y'].append(threshold)

            if ie_price < threshold:
                simulator.sell(ie_price)
                direction *= -1
                # state = 'estimate_slope'
                slope_start_x = idx

    p = np.array(runner.ie_prices)
    f, (ax1, ax2) = plt.subplots(2, 1, sharex='all', gridspec_kw={'height_ratios': [3, 1]})
    # ax1 = plt.subplot(2, 1, 1)
    # plt.plot(times, prices, label=f'price')
    # plt.plot(times, asks, label=f'ask')
    # plt.plot(times, bids, label=f'bid')
    # plt.plot(runner.os_times, runner.os_prices, label=f'OS')
    # plt.scatter(runner.os_times, runner.os_prices, label=f'OS', s=5 ** 2)
    # plt.scatter(runner.dc_times, runner.dc_prices, label=f'DC', s=7 ** 2)
    # plt.scatter(runner.ie_times, runner.ie_prices, label=f'IE', s=5 ** 2)

    # plt.scatter(x, runner.dc_prices, label=f'DC', s=7 ** 2)
    ax1.scatter(x, runner.ie_prices, label=f'IE', s=2 ** 2, c='green')
    for smooth_period in smooths:
        smooth = smooths[smooth_period]
        ax1.plot(x, smooth, label=f'Smooth {smooth_period}')

    for slope in slopes:
        # print(slope['x'])
        # print(slope['y'])
        ax1.plot(slope['x'], slope['y'])

    ax1.plot(thresholds['x'], thresholds['y'], label=f'Threshold')

    ax1.plot()
    ax1.grid(True)
    ax1.set_yscale('log')

    ax2.plot(x, values, label=f'Value')

    # plt.scatter(runner.ie_times, runner.ie_min_asks, label=f'Min Asks', s=5 ** 2)
    # plt.scatter(runner.ie_times, runner.ie_max_bids, label=f'Max Bids', s=5 ** 2)

    plt.legend()

    plt.show()
