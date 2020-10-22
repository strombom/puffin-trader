
import sys
from logging import _addHandlerRef

import numpy as np
from enum import Enum
from datetime import timedelta
import matplotlib.pyplot as plt
sys.path.append("../Common")

#import matplotlib.pyplot as plt
from OrderBook import make_order_books, order_books_to_csv
from misc import read_agg_ticks


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
        self.direction = Direction.up
        self.extreme_price = order_book.ask
        self.extreme_timestamp = order_book.timestamp
        self.delta_price = order_book.ask * (1 - self.delta)
        self.dc_times = []
        self.dc_prices = []
        self.os_times = []
        self.os_prices = []

    def step(self, order_book):
        if self.direction == Direction.up:
            if order_book.ask > self.extreme_price:
                self.extreme_price = order_book.ask
                self.extreme_timestamp = order_book.timestamp
                self.delta_price = order_book.ask * (1 - self.delta)

            elif order_book.bid < self.delta_price:
                self._append(order_book.timestamp)
                self.direction = Direction.down
                self.delta_price = order_book.bid * (1 + self.delta)
                return RunnerEvent.change_down
        else:
            if order_book.bid < self.extreme_price:
                self.extreme_price = order_book.bid
                self.extreme_timestamp = order_book.timestamp
                self.delta_price = order_book.bid * (1 + self.delta)

            elif order_book.ask > self.delta_price:
                self._append(order_book.timestamp)
                self.direction = Direction.up
                self.delta_price = order_book.ask * (1 - self.delta)
                return RunnerEvent.change_up

        return RunnerEvent.nothing, 0, 0, 0, 0

    def _append(self, dc_timestamp):
        self.dc_times.append(dc_timestamp)
        self.dc_prices.append(self.delta_price)
        self.os_times.append(self.extreme_timestamp)
        self.os_prices.append(self.extreme_price)


if __name__ == '__main__':
    order_books = make_order_books(None, None)
    if order_books is None:
        agg_ticks = read_agg_ticks('C:/development/github/puffin-trader/tmp/agg_ticks.csv')
        order_books = make_order_books(agg_ticks, timedelta(minutes=1))

    print(order_books[0])
    print(order_books[-1])

    deltas = [0.0027, 0.0033, 0.0039, 0.0047, 0.0056, 0.0068, 0.0082, 0.010, 0.012, 0.015]
    runners = []
    for delta in deltas:
        runners.append(Runner(delta=delta, order_book=order_books[0]))

    for order_book in order_books:
        for runner in runners:
            runner.step(order_book)

    x_prices = []
    y_prices_a = []
    y_prices_b = []
    for order_book in order_books:
        x_prices.append(order_book.timestamp)
        y_prices_a.append(order_book.ask)
        y_prices_b.append(order_book.bid)
    plt.plot(x_prices, y_prices_a, color='blue', alpha=0.3)
    plt.plot(x_prices, y_prices_b, color='blue', alpha=0.3)

    #for idx, runner in enumerate(runners):
    #    #plt.plot(runner.os_times, runner.os_prices)
    #    plt.plot(runner.dc_times, runner.dc_prices)
    #    if runner == 4:
    #        break

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
