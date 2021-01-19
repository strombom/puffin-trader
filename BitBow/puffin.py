import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from enum import IntEnum
from scipy import stats
from datetime import datetime, timezone

from Indicators.supersmoother import SuperSmoother
from IntrinsicTime.runner import Direction
from rainbow import rainbow_indicator_load_params, rainbow_indicator


class PositionDirection(IntEnum):
    long = 1
    hedge = 0
    short = -1


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

    runner.ie_prices = np.array(runner.ie_prices)[0:1000]
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
        # print('slope', r.slope)
        return c_x, c_x * r.slope + r.intercept


    plot_events = {
        PositionDirection.long:  {'x': [], 'y': []},
        PositionDirection.hedge: {'x': [], 'y': []},
        PositionDirection.short: {'x': [], 'y': []}
    }

    first_idx = 100
    simulator = Simulator()
    direction = PositionDirection.short
    state = 'estimate_slope'
    puffin_price = runner.ie_prices[0]
    slope = None
    slope_start_x = first_idx - 20
    threshold = puffin_price * (1 + 2 * delta)
    thresholds = {'x': [], 'y': []}
    slopes = []
    values = []


    def find_best_fit(idx: int):
        pass


    find_best_fit(first_idx)

    for idx, ie_price in enumerate(runner.ie_prices[:first_idx]):
        values.append(simulator.get_value(ie_price))
        plot_events[PositionDirection.hedge]['x'].append(idx)
        plot_events[PositionDirection.hedge]['y'].append(ie_price)

    for idx in range(first_idx, runner.ie_prices.shape[0]):
        ie_price = runner.ie_prices[idx]
        values.append(simulator.get_value(ie_price))
        slope_start_x = max(slope_start_x, idx - 20)

        if idx - slope_start_x >= 10:
            slope_x, slope_y = estimate_slope(slope_start_x, idx)
            # slopes.append({'x': slope_x, 'y': slope_y})

        plot_events[direction]['x'].append(idx)
        plot_events[direction]['y'].append(ie_price)

        if direction == PositionDirection.short:
            threshold = slope_y[-1] * (1 + 2 * delta)
            thresholds['x'].append(idx)
            thresholds['y'].append(threshold)

            if ie_price > threshold:
                simulator.buy(ie_price)
                direction *= -1
                # state = 'estimate_slope'
                slope_start_x = idx

        elif direction == PositionDirection.long:
            threshold = slope_y[-1] * (1 - 2 * delta)
            thresholds['x'].append(idx)
            thresholds['y'].append(threshold)

            if ie_price < threshold:
                simulator.sell(ie_price)
                direction *= -1
                # state = 'estimate_slope'
                slope_start_x = idx

    f, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1,
                                      sharex='col',
                                      gridspec_kw={'height_ratios': [4, 1, 1]},
                                      figsize=(10, 9))
    ax1.grid(which='minor', alpha=0.3)
    ax1.grid(which='major', alpha=0.3)
    ax1.set_yscale('log', base=10)
    ax1.set_xlim([first_idx - 50, first_idx + 200])
    fmt = FormatStrFormatter('%.0f')
    ax1.yaxis.set_major_formatter(fmt)
    ax1.yaxis.set_minor_formatter(fmt)

    ax1.scatter(thresholds['x'], thresholds['y'], marker='_', c='xkcd:gold', label=f'Threshold')

    for direction, color, label in ((PositionDirection.long, 'xkcd:green', 'Long'),
                                    (PositionDirection.hedge, 'xkcd:light blue', 'Hedge'),
                                    (PositionDirection.short, 'xkcd:red', 'Short')):
        ax1.scatter(plot_events[direction]['x'], plot_events[direction]['y'], label=label, s=2 ** 2, c=color)

    ax2.grid(True)
    ax3.grid(True)

    ax1.legend(loc='upper left')

    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    plt.tight_layout()
    plt.show()
    plt.get_current_fig_manager().toolbar.pan()

    """
    # formatter = ScalarFormatter()
    # ax1.yaxis.set_major_formatter(formatter)
    # ax1.ticklabel_format(axis='y', style='plain', useOffset=False)
    # formatter.set_scientific(False)

    # plt.scatter(runner.os_times, runner.os_prices, label=f'OS', s=5 ** 2)
    # plt.scatter(runner.dc_times, runner.dc_prices, label=f'DC', s=7 ** 2)
    # plt.scatter(runner.ie_times, runner.ie_prices, label=f'IE', s=5 ** 2)

    for slope in slopes:
        # print(slope['x'])
        # print(slope['y'])
        ax1.plot(slope['x'], slope['y'])
    """
