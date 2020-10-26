
import sys
from logging import _addHandlerRef

import numpy as np
from enum import Enum
from datetime import timedelta
#import matplotlib.pyplot as plt
#import matplotlib
sys.path.append("../Common")

#import matplotlib.pyplot as plt
from OrderBook import make_order_books, order_books_to_csv
from misc import read_agg_ticks
from plot import Plot


class Direction(Enum):
    up = 1
    down = -1


class RunnerEvent(Enum):
    change_up = 1
    nothing = 0
    change_down = -1


class Runner:
    def __init__(self, delta, order_book):
        self.delta = delta
        self.delta_up = delta * 1.0
        self.delta_down = delta * 1.0
        self.direction = Direction.up
        self.extreme_price = order_book.ask
        self.extreme_timestamp = order_book.timestamp
        self.delta_price = order_book.ask * (1 - self.delta_down)
        self.ie_price = order_book.ask * (1 + self.delta_up)
        self.dc_times = []
        self.dc_prices = []
        self.os_times = []
        self.os_prices = []
        self.ie_times = []
        self.ie_prices = []

    def step(self, order_book):
        if self.direction == Direction.up:
            if order_book.ask > self.extreme_price:
                self.extreme_price = order_book.ask
                self.extreme_timestamp = order_book.timestamp
                self.delta_price = order_book.ask * (1 - self.delta_down)

                if order_book.ask > self.ie_price:
                    self.ie_times.append(order_book.timestamp.timestamp())
                    self.ie_prices.append(self.ie_price)
                    self.ie_price = order_book.ask * (1 + self.delta_up)

            elif order_book.bid < self.delta_price:
                self._append(order_book.timestamp)
                self.direction = Direction.down
                self.delta_price = order_book.bid * (1 + self.delta_up)
                self.ie_price = order_book.bid * (1 - self.delta_down)
                return RunnerEvent.change_down

        else:
            if order_book.bid < self.extreme_price:
                self.extreme_price = order_book.bid
                self.extreme_timestamp = order_book.timestamp
                self.delta_price = order_book.bid * (1 + self.delta_up)

                if order_book.bid < self.ie_price:
                    self.ie_times.append(order_book.timestamp.timestamp())
                    self.ie_prices.append(self.ie_price)
                    self.ie_price = order_book.ask * (1 - self.delta_down)

            elif order_book.ask > self.delta_price:
                self._append(order_book.timestamp)
                self.direction = Direction.up
                self.delta_price = order_book.ask * (1 - self.delta_down)
                self.ie_price = order_book.ask * (1 + self.delta_up)
                return RunnerEvent.change_up

        return RunnerEvent.nothing, 0, 0, 0, 0

    def _append(self, dc_timestamp):
        #self.ie_times.append(self.extreme_timestamp.timestamp())
        #self.ie_prices.append(self.extreme_price)
        self.ie_times.append(dc_timestamp.timestamp())
        self.ie_prices.append(self.delta_price)
        self.os_times.append(self.extreme_timestamp.timestamp())
        self.os_prices.append(self.extreme_price)
        self.dc_times.append(dc_timestamp.timestamp())
        self.dc_prices.append(self.delta_price)


if __name__ == '__main__':
    order_books = make_order_books(None, None)
    if order_books is None:
        agg_ticks = read_agg_ticks('C:/development/github/puffin-trader/tmp/agg_ticks.csv')
        order_books = make_order_books(agg_ticks, timedelta(minutes=1))

    order_books = order_books[:10000]
    print(order_books[0])
    print(order_books[-1])

    #deltas = [0.0027, 0.0033, 0.0039, 0.0047, 0.0056, 0.0068, 0.0082, 0.010, 0.012, 0.015, 0.018, 0.022, 0.027, 0.033, 0.039, 0.047]
    # 0.00051, 0.00056, 0.00062, 0.00068, 0.00075, 0.00082, 0.00091, 0.0010, 0.0011, 0.0012, 0.0013, 0.0015, 0.0016, 0.0018, 0.0020, 0.0022, 0.0024, 0.0027, 0.0030,
    deltas = [0.0033, 0.0036, 0.0039, 0.0043, 0.0047, 0.0051, 0.0056, 0.0062, 0.0068, 0.0075, 0.0082, 0.0091, 0.010, 0.011, 0.012, 0.013, 0.015, 0.018, 0.020] #] #, 0.022, 0.024, 0.027, 0.030, 0.033, 0.036, 0.039, 0.043, 0.047, 0.051]
    delta_clock = 0.001

    #deltas = [0.005]
    runners = []
    for delta in deltas:
        runners.append(Runner(delta=delta, order_book=order_books[0]))
    runner_clock = Runner(delta=delta_clock, order_book=order_books[0])

    volatility_period = timedelta(hours=1)
    current_time = order_books[0].timestamp
    print("current_time", current_time)

    for order_book in order_books:
        runner_clock.step(order_book)
        for runner in runners:
            event = runner.step(order_book)

    """
    x_prices, y_prices_a, y_prices_b = [], [], []
    for order_book in order_books:
        x_prices.append(order_book.timestamp.timestamp())
        y_prices_a.append(order_book.ask)
        y_prices_b.append(order_book.bid)
    plt.plot(x_prices, y_prices_a, color='green', alpha=0.4)
    plt.plot(x_prices, y_prices_b, color='green', alpha=0.4)
    for runner in [runners[0]]:
        plt.scatter(runner.ie_times, runner.ie_prices, color='orange', s=100)
        plt.plot(runner.os_times, runner.os_prices, color='green')
        plt.scatter(runner.os_times, runner.os_prices, color='green')
        plt.scatter(runner.dc_times, runner.dc_prices, color='red')

    plt.scatter(runner_clock.ie_times, runner_clock.ie_prices, color='blue', s=20)
    #plt.show()
    """

    target_direction = np.zeros((len(deltas), len(runner_clock.ie_times)))
    measured_direction = np.zeros((len(deltas), len(runner_clock.ie_times)))
    TMV = np.zeros((len(deltas), len(runner_clock.ie_times)))
    print(TMV.shape)
    print(len(order_books))
    #quit()

    # Target direction
    for idx_runner, runner in enumerate(runners):
        direction = Direction.up
        idx_os = 0
        for idx_clock, timestamp in enumerate(runner_clock.ie_times):
            while idx_os < len(runner.os_times) and runner.os_times[idx_os] < timestamp:
                idx_os += 1
                if direction == Direction.up:
                    direction = Direction.down
                else:
                    direction = Direction.up
            if idx_os >= len(runner.os_times):
                break
            if direction == Direction.up:
                target_direction[idx_runner, idx_clock] = idx_runner * 1 + 1
            else:
                target_direction[idx_runner, idx_clock] = idx_runner * 1 + 0

    # Measured direction
    for idx_runner, runner in enumerate(runners):
        direction = Direction.up
        idx_dc = 0
        for idx_clock, timestamp in enumerate(runner_clock.ie_times):
            while idx_dc < len(runner.dc_times) and runner.dc_times[idx_dc] < timestamp:
                idx_dc += 1
                if direction == Direction.up:
                    direction = Direction.down
                else:
                    direction = Direction.up
            if idx_dc >= len(runner.dc_times):
                break
            if direction == Direction.up:
                measured_direction[idx_runner, idx_clock] = idx_runner * 1 + 1
            else:
                measured_direction[idx_runner, idx_clock] = idx_runner * 1 + 0

    x = np.arange(len(runner_clock.ie_times))

    plot = Plot()
    plot.plot((x, runner_clock.ie_prices, target_direction, measured_direction))
    plot.show()
    plot.print('abc')
    while True:
        cmd, payload = plot.get()
        if cmd == 'quit':
            break
        elif cmd == 'x':
            print("got x", payload)
    plot.shutdown()

    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(x, runner_clock.ie_prices)

    for idx in range(len(runners)):
        ax2.fill_between(x, 1 * idx, target_direction[idx])
    for idx in range(len(runners)):
        ax3.fill_between(x, 1 * idx, measured_direction[idx])

    fig.tight_layout()
    plt.show()
    """

    quit()




    """
    volatilities = []
    events = []
    for order_book in order_books:
        for runner in runners:
            event = runner.step(order_book)
            if event != RunnerEvent.nothing:
                events.append(order_book)
        current_time = events[-1].timestamp
        count = 0
        for idx in range(len(events)):
            if events[idx].timestamp > current_time - volatility_period:
                break
            count += 1
        if count > 0:
            if len(events) > count:
                events = events[count:]
            else:
                events = []
        volatilities.append(len(events))
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)











    """
    for idx in range(1, len(dc_timestamps)):

        d = os_timestamps[idx] - os_timestamps[idx - 1]
        dt = os_timestamps[idx - 1] - dc_timestamps[idx - 1]
        ds.append(d)

        #print("time", d, dt, dt / d)

        h = os_prices[idx] - os_prices[idx - 1]
        dh = os_prices[idx - 1] - dc_prices[idx - 1]
        hs.append(h)

        if d.seconds > 0 and dh != 0:
            dts.append(dt.seconds / d.seconds)
            dhs.append(h / dh)

        #print("price", h, dh, dh / h)
    """






    x_prices, y_prices_a, y_prices_b = [], [], []
    for order_book in order_books:
        x_prices.append(order_book.timestamp.timestamp())
        y_prices_a.append(order_book.ask)
        y_prices_b.append(order_book.bid)
    ax1.plot(x_prices, y_prices_a, color='green', alpha=0.4)
    ax1.plot(x_prices, y_prices_b, color='green', alpha=0.4)

    #for runner in [runners[0]]:
    #    ax1.plot(runner.ie_times, runner.ie_prices, color='green')
    #    ax1.scatter(runner.ie_times, runner.ie_prices, color='green')
    #    ax1.plot(runner.os_times, runner.os_prices, color='red')
    #    ax1.scatter(runner.dc_times, runner.dc_prices, color='red')

    ax1.plot(runner_clock.ie_times, runner_clock.ie_prices, color='green')
    ax1.scatter(runner_clock.ie_times, runner_clock.ie_prices, color='green')
    #ax1.plot(runner.os_times, runner.os_prices, color='red')
    #ax1.scatter(runner.dc_times, runner.dc_prices, color='red')

    #ax2.plot(x_prices, volatilities, color='green')
    #plt.show()
    #quit()


    for idx, runner in enumerate(runners):
        #if idx == 3:
        #    break
        dc_x, dc_y = np.array(runner.dc_times), np.array(runner.dc_prices)
        dc_y = np.ones(len(dc_x)) * idx
        points = np.array([dc_x, dc_y]).transpose().reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        os_y = np.array(runner.os_prices)
        directions = os_y[1:] - os_y[:-1]

        color_map = matplotlib.colors.ListedColormap(('red', 'green'))
        boundary_norm = matplotlib.colors.BoundaryNorm([-1000, 0, 1000], color_map.N)
        line_collection = matplotlib.collections.LineCollection(segments, cmap=color_map, norm=boundary_norm)
        line_collection.set_array(directions)
        line_collection.set_linewidth(45)

        line = ax2.add_collection(line_collection)
        #fig.colorbar(line, ax=ax2)

        ax2.set_xlim(dc_x[0], dc_x[-1])
        ax2.set_ylim(-1, idx + 1)

        #print(color_map)
        #ys = np.ones(len(xs))

        #¤segments =

        #ax2.plot(xs, ys)


        #break

    #    #plt.plot(runner.os_times, runner.os_prices)
    #    plt.plot(runner.dc_times, runner.dc_prices)
    #    if runner == 4:
    #        break



    fig.tight_layout()
    plt.show()

    quit()


"""
    x_prices = []
    y_pricesa = []
    y_pricesb = []
    dc_timestamps = []
    dc_prices = []
    os_timestamps = []
    os_prices = []

    for order_book in order_books:
        x_prices.append(order_book.timestamp)
        y_pricesa.append(order_book.ask)
        y_pricesb.append(order_book.bid)

        event runner.step(order_book)

        if event != RunnerEvent.nothing:
            dc_timestamps.append(dc_timestamp)
            dc_prices.append(dc_price)
            os_timestamps.append(os_timestamp)
            os_prices.append(os_price)

    ds = []
    dts = []
    hs = []
    dhs = []

    for idx in range(1, len(dc_timestamps)):

        d = os_timestamps[idx] - os_timestamps[idx - 1]
        dt = os_timestamps[idx - 1] - dc_timestamps[idx - 1]
        ds.append(d)

        #print("time", d, dt, dt / d)

        h = os_prices[idx] - os_prices[idx - 1]
        dh = os_prices[idx - 1] - dc_prices[idx - 1]
        hs.append(h)

        if d.seconds > 0 and dh != 0:
            dts.append(dt.seconds / d.seconds)
            dhs.append(h / dh)

        #print("price", h, dh, dh / h)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)

    ax1.scatter(dhs[0::2], dts[0::2])
    ax1.scatter(dhs[1::2], dts[1::2])
    #ax2.scatter(dhs)
    plt.show()
    quit()


    plt.plot(x_prices, y_pricesa)
    plt.plot(x_prices, y_pricesb)
    plt.plot(os_timestamps, os_prices)
    plt.scatter(dc_timestamps, dc_prices, color='red')
    plt.show()


"""
