
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from enum import IntEnum
from scipy import stats

from BinanceSim.binance_simulator import BinanceSimulator


class PositionDirection(IntEnum):
    long = 1
    hedge = 0
    short = -1


class Slope:
    max_slope_length = 70

    def __init__(self, ie_prices, end_idx):
        self.ie_prices = ie_prices
        slope_x, slope_y = self.find_best_fit(end_idx - self.max_slope_length, end_idx)
        dx, dy = slope_x[-1] - slope_x[0], 1000 * (slope_y[-1] - slope_y[0]) / ie_price
        self.angle = dy / dx
        self.length = slope_x[-1] - slope_x[0]
        self.x = slope_x
        self.y = slope_y

    def estimate_slope(self, start, stop):
        c_x = np.arange(start, stop + 1)
        r = stats.linregress(c_x, self.ie_prices[start:stop + 1])
        return c_x, c_x * r.slope + r.intercept

    def find_best_fit(self, idx_start: int, idx_end: int):
        xs, ys = [], []
        for l in range(idx_end + 1 - idx_start, 10, -1):
            slope_x, slope_y = self.estimate_slope(idx_end - l, idx_end)
            max_d = 0
            for x, y in zip(slope_x, slope_y):
                d = abs(self.ie_prices[x] - y)
                max_d = max(max_d, d)
            xs.append(slope_x[0])
            ys.append(max_d / slope_y[0] * 100 - l / 75)

        min_x_idx = ys.index(min(ys))
        return self.estimate_slope(xs[min_x_idx], idx_end)


class Plotter:
    def __init__(self):
        self.events = {
            PositionDirection.long:  {'x': [], 'y': []},
            PositionDirection.hedge: {'x': [], 'y': []},
            PositionDirection.short: {'x': [], 'y': []}
        }
        self.angles = {'x': [], 'y': []}
        self.values = {'x': [], 'y': []}
        self.thresholds = {'x': [], 'y': []}

    def append_event(self, event_direction: PositionDirection, event_idx: int, event_price: float):
        self.events[PositionDirection.hedge]['x'].append(event_idx)
        self.events[PositionDirection.hedge]['y'].append(event_price)

    def append_angle(self, angle_idx: int, angle: float):
        self.angles['x'].append(angle_idx)
        self.angles['y'].append(angle)

    def append_value(self, value_idx: int, value_price: float):
        self.values['x'].append(value_idx)
        self.values['y'].append(value_price)

    def append_threshold(self, threshold_idx: int, threshold: float):
        self.thresholds['x'].append(idx)
        self.thresholds['y'].append(threshold)


class Position:
    def __init__(self, direction: PositionDirection, plotter: Plotter):
        self.direction = direction
        self.plotter = plotter

    def step(self, mark_price: float) -> bool:
        # threshold_delta = (1.6 + 0.8 * (slope_len - 10) / 70) * delta
        threshold_delta = 1.85 * delta

        if abs(slope.angle) > 2 and slope.length > 30:
            threshold_delta *= 0.9
        # elif abs(slope_angle) > 1:
        #     threshold_delta *= 1.1

        # threshold_delta *= (1 - max(0.5, min(1, abs(anglediff))) / 2)

        make_trade = False

        if position.direction == PositionDirection.short:
            threshold = slope.y[-1] * (1 + threshold_delta)
            plotter.append_threshold(idx, threshold)

            if ie_price > threshold or \
                    (ie_price > slope.y[-1] and slope.angle > angle_threshold and slope.angle > prev_slope_angle) or \
                    (ie_price > slope.y[-1] and slope.angle > 0 and slope.length > 20):
                make_trade = True

        elif position.direction == PositionDirection.long:
            threshold = slope.y[-1] * (1 - threshold_delta)
            plotter.append_threshold(idx, threshold)

            if ie_price < threshold or \
                    (ie_price < slope.y[-1] and slope.angle < -angle_threshold and slope.angle < prev_slope_angle) or \
                    (ie_price < slope.y[-1] and slope.angle < 0 and slope.length > 20):
                make_trade = True

        # if abs(anglediff) > 1.0:
        #     threshold_delta *= 1 - 0.5
        # elif abs(anglediff) > 0.7:
        #     threshold_delta *= 1 - 0.3
        # elif abs(anglediff) > 0.4:
        #     threshold_delta *= 1 - 0.2

        return make_trade


if __name__ == '__main__':
    with open(f"cache/intrinsic_time_runner.pickle", 'rb') as f:
        delta, runner = pickle.load(f)

    runner.ie_prices = np.array(runner.ie_prices)[0:500]
    x = np.arange(runner.ie_prices.shape[0])

    first_idx = 96
    simulator = BinanceSimulator(initial_usdt=0.0, initial_btc=1.0, max_leverage=2, mark_price=runner.ie_prices[0], initial_leverage=0.0)
    plotter = Plotter()
    position = Position(direction=PositionDirection.long, plotter=plotter)

    slopes = []

    angle_threshold = 0.2
    annotations = []

    for idx, ie_price in enumerate(runner.ie_prices[:first_idx]):
        plotter.append_event(PositionDirection.hedge, idx, ie_price)
        slopes.append(None)

    previous_trade_value = simulator.get_value_usdt(mark_price=runner.ie_prices[first_idx])

    for idx in range(first_idx, runner.ie_prices.shape[0]):
        ie_price = runner.ie_prices[idx]
        plotter.append_value(idx, simulator.get_value_usdt(mark_price=ie_price))

        slope = Slope(runner.ie_prices, idx)
        slopes.append(slope)

        prev_slope_angle = 0
        if slopes[-2] is not None:
            prev_slope_angle = slopes[-2].angle

        plotter.append_angle(idx, slope.angle)

        make_trade = position.step(ie_price)

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

            annotations.append({
                'x': idx,
                'y': ie_price,
                'direction': 'short',
                'profit': profit
            })

        plotter.append_event(position.direction, idx, ie_price)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1,
                                        sharex='col',
                                        gridspec_kw={'height_ratios': [4, 1, 1]},
                                        figsize=(10, 10))
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    ax1.grid(which='minor', alpha=0.3)
    ax1.grid(which='major', alpha=0.3)
    ax1.set_yscale('log', base=10)
    ax1.set_xlim([first_idx - 25, first_idx + 100])
    fmt = FormatStrFormatter('%.0f')
    ax1.yaxis.set_major_formatter(fmt)
    ax1.yaxis.set_minor_formatter(fmt)

    ax1.scatter(plotter.thresholds['x'], plotter.thresholds['y'], marker='_', c='xkcd:gold', label=f'Threshold')

    for direction, color, label in ((PositionDirection.long, 'xkcd:green', 'Long'),
                                    (PositionDirection.hedge, 'xkcd:light blue', 'Hedge'),
                                    (PositionDirection.short, 'xkcd:red', 'Short')):
        ax1.scatter(plotter.events[direction]['x'], plotter.events[direction]['y'], label=label, s=2 ** 2, c=color)

    # for slope in slopes:
    #     ax1.plot(slope['x'], slope['y'])
    slope_plot = ax1.plot([0], [runner.ie_prices[0]], c='xkcd:hot pink')[0]

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
    ax2.plot(plotter.angles['x'], plotter.angles['y'], label=f'Angle')
    angle_diffs = np.array(plotter.angles['y'])[1:] - np.array(plotter.angles['y'])[:-1]
    ax2.plot(plotter.angles['x'][1:], angle_diffs, label=f'Angle diff')
    ax2.legend(loc='upper left')

    ax3.grid(True)
    ax3.plot(plotter.values['x'], plotter.values['y'], label=f'Value')
    ax3.legend(loc='upper left')
    # ax3.set_ylim([0, 3])


    def on_mouse_move(event):
        if event.xdata is None or event.xdata < 70:
            return
        slope_idx = int(event.xdata + 0.5)
        # (mind_x, mind_y), (slope_x, slope_y) = find_best_fit(x - 70, x)
        # mindicator.set_data(mind_x, mind_y)
        if 0 <= slope_idx < len(slopes) and slopes[slope_idx] is not None:
            slope_plot.set_data(slopes[slope_idx].x, slopes[slope_idx].y)
            plt.draw()
            # fig.canvas.blit(ax1.bbox)

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

    """
    smooths = {}
    smooth_periods = [8]
    for smooth_period in smooth_periods:
        smooth = []
        smoother = SuperSmoother(period=smooth_period, initial_value=runner.ie_prices[0])
        for price in runner.ie_prices:
            smooth.append(smoother.append(price))
        smooths[smooth_period] = smooth
    """
