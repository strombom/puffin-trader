import math
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
    def __init__(self, btc, usd):
        self.btc = btc
        self.usd = usd
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

    runner.ie_prices = np.array(runner.ie_prices)[0:500]
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
        c_x = np.arange(start, stop + 1)
        r = stats.linregress(c_x, runner.ie_prices[start:stop + 1])
        return c_x, c_x * r.slope + r.intercept


    plot_events = {
        PositionDirection.long:  {'x': [], 'y': []},
        PositionDirection.hedge: {'x': [], 'y': []},
        PositionDirection.short: {'x': [], 'y': []}
    }

    first_idx = 96
    simulator = Simulator(btc=1, usd=0)
    direction = PositionDirection.long
    state = 'estimate_slope'
    puffin_price = runner.ie_prices[0]
    slope = None
    slope_start_x = first_idx - 20
    threshold = puffin_price * (1 + 2 * delta)
    thresholds = {'x': [], 'y': []}
    slopes = []
    values = {'x': [], 'y': []}
    angles = {'x': [], 'y': []}
    anglediffs = {'x': [], 'y': []}
    angle_threshold = 0.2
    annotations = []


    def find_best_fit(idx_start: int, idx_end: int):
        xs, ys = [], []
        for l in range(idx_end + 1 - idx_start, 10, -1):
            slope_x, slope_y = estimate_slope(idx_end - l, idx_end)
            max_d = 0
            for x, y in zip(slope_x, slope_y):
                d = abs(runner.ie_prices[x] - y)
                max_d = max(max_d, d)
            xs.append(slope_x[0])
            ys.append(max_d / slope_y[0] * 100 - l / 75)

        min_x_idx = ys.index(min(ys))
        # print(xs[min_x_idx])
        slope = estimate_slope(xs[min_x_idx], idx_end)
        mindicator = (xs, ys)
        return mindicator, slope

    # find_best_fit(first_idx - 70, first_idx)
    # find_best_fit(100, 150)

    # for s in range(25, 500, 20):
    #     find_best_fit(s, s + 70)

    for idx, ie_price in enumerate(runner.ie_prices[:first_idx]):
        plot_events[PositionDirection.hedge]['x'].append(idx)
        plot_events[PositionDirection.hedge]['y'].append(ie_price)

    prev_slope_angle = 0
    slope_angle = 0
    previous_trade_value = simulator.get_value(runner.ie_prices[first_idx])
    confirm_negative_slope = False

    for idx in range(first_idx, runner.ie_prices.shape[0]):
        ie_price = runner.ie_prices[idx]
        values['x'].append(idx)
        values['y'].append(simulator.get_value(ie_price))
        slope_start_x = max(slope_start_x, idx - 20)

        if idx == 242:
            a = 1

        (mind_x, mind_y), (slope_x, slope_y) = find_best_fit(idx - 70, idx)
        slope_len = slope_x[-1] - slope_x[0]
        # threshold_delta = (1.6 + 0.8 * (slope_len - 10) / 70) * delta
        threshold_delta = 1.85 * delta

        dx, dy = slope_x[-1] - slope_x[0], 1000 * (slope_y[-1] - slope_y[0]) / ie_price
        # angle = math.atan(dy / dx)
        prev_slope_angle = slope_angle
        slope_angle = dy / dx
        angles['x'].append(idx)
        angles['y'].append(slope_angle)

        if abs(slope_angle) > 2 and slope_len > 30:
            threshold_delta *= 0.9
        # elif abs(slope_angle) > 1:
        #     threshold_delta *= 1.1

        anglediff = slope_angle - prev_slope_angle
        anglediffs['x'].append(idx)
        anglediffs['y'].append(anglediff)

        # threshold_delta *= (1 - max(0.5, min(1, abs(anglediff))) / 2)

        # slope_x, slope_y = estimate_slope(slope_start_x, idx)
        # slopes.append({'x': slope_x, 'y': slope_y})

        plot_events[direction]['x'].append(idx)
        plot_events[direction]['y'].append(ie_price)

        make_trade = False

        if abs(anglediff) > 1.0:
            threshold_delta *= 1 - 0.5
        elif abs(anglediff) > 0.7:
            threshold_delta *= 1 - 0.3
        elif abs(anglediff) > 0.4:
            threshold_delta *= 1 - 0.2

        if direction == PositionDirection.short:
            threshold = slope_y[-1] * (1 + threshold_delta)
            thresholds['x'].append(idx)
            thresholds['y'].append(threshold)

            if ie_price > threshold or \
                    (ie_price > slope_y[-1] and slope_angle > angle_threshold and slope_angle > prev_slope_angle) or \
                    (slope_len > 20 and slope_angle > 0 and ie_price > slope_y[-1]):
                make_trade = True

        elif direction == PositionDirection.long:
            threshold = slope_y[-1] * (1 - threshold_delta)
            thresholds['x'].append(idx)
            thresholds['y'].append(threshold)

            if ie_price < threshold or \
                    (ie_price < slope_y[-1] and slope_angle < -angle_threshold and slope_angle < prev_slope_angle) or \
                    (slope_len > 20 and slope_angle < 0 and ie_price < slope_y[-1]):
                make_trade = True

        if make_trade:
            if direction == PositionDirection.short:
                simulator.buy(ie_price)
            else:
                simulator.sell(ie_price)
            direction *= -1
            slope_start_x = idx
            made_trade = True

            value_after = simulator.get_value(ie_price)
            profit = (value_after - previous_trade_value) / previous_trade_value
            previous_trade_value = value_after

            annotations.append({
                'x': idx,
                'y': ie_price,
                'direction': 'short',
                'profit': profit
            })

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1,
                                        sharex='col',
                                        gridspec_kw={'height_ratios': [4, 1, 1]},
                                        figsize=(10, 10))
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    ax1.grid(which='minor', alpha=0.3)
    ax1.grid(which='major', alpha=0.3)
    ax1.set_yscale('log', base=10)
    ax1.set_xlim([first_idx - 100, first_idx + 100])
    fmt = FormatStrFormatter('%.0f')
    ax1.yaxis.set_major_formatter(fmt)
    ax1.yaxis.set_minor_formatter(fmt)

    ax1.scatter(thresholds['x'], thresholds['y'], marker='_', c='xkcd:gold', label=f'Threshold')

    for direction, color, label in ((PositionDirection.long, 'xkcd:green', 'Long'),
                                    (PositionDirection.hedge, 'xkcd:light blue', 'Hedge'),
                                    (PositionDirection.short, 'xkcd:red', 'Short')):
        ax1.scatter(plot_events[direction]['x'], plot_events[direction]['y'], label=label, s=2 ** 2, c=color)

    # for slope in slopes:
    #     ax1.plot(slope['x'], slope['y'])
    slope = ax1.plot([0], [runner.ie_prices[0]], c='xkcd:hot pink')[0]

    for annotation in annotations:
        if annotation['profit'] > 0:
            color = 'xkcd:green'
        else:
            color = 'xkcd:red'
        ax1.annotate(f'{annotation["profit"] * 100:.2f}',
                     xy=(annotation['x'], annotation['y'] * 1.001),
                     xytext=(annotation['x'], annotation['y'] * 1.05),
                     color='xkcd:black',
                     bbox={'boxstyle': 'round', 'fc': color, 'alpha': 0.5},
                     arrowprops={'arrowstyle': '->', 'color': 'cornflowerblue'},
                     horizontalalignment='center')

    ax1.legend(loc='upper left')

    ax2.grid(True)
    # ax2.set_ylim([0, 3])
    # mindicator = ax2.plot([1], [2], c='red')[0]
    ax2.plot(angles['x'], angles['y'], label=f'Angle')
    ax2.plot(anglediffs['x'], anglediffs['y'], label=f'Angle diff')
    ax2.legend(loc='upper left')

    ax3.grid(True)
    ax3.plot(values['x'], values['y'], label=f'Value')
    ax3.legend(loc='upper left')
    # ax3.set_ylim([0, 3])


    def on_mouse_move(event):
        if event.xdata is None or event.xdata < 70:
            return
        x = int(event.xdata + 0.5)
        (mind_x, mind_y), (slope_x, slope_y) = find_best_fit(x - 70, x)
        # mindicator.set_data(mind_x, mind_y)
        slope.set_data(slope_x, slope_y)
        plt.draw()

    plt.connect('motion_notify_event', on_mouse_move)
    plt.show()
    # plt.get_current_fig_manager().toolbar.pan()

    """
    # formatter = ScalarFormatter()
    # ax1.yaxis.set_major_formatter(formatter)
    # ax1.ticklabel_format(axis='y', style='plain', useOffset=False)
    # formatter.set_scientific(False)

    # plt.scatter(runner.os_times, runner.os_prices, label=f'OS', s=5 ** 2)
    # plt.scatter(runner.dc_times, runner.dc_prices, label=f'DC', s=7 ** 2)
    # plt.scatter(runner.ie_times, runner.ie_prices, label=f'IE', s=5 ** 2)
    """
