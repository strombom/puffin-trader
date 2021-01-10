
import pickle
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from Common.AggTicks import read_agg_ticks
from Common.Misc import string_to_datetime
from Common.OrderBook import make_order_books
from IntrinsicTime.runner import Runner


if __name__ == "__main__":

    ignore_date_ranges = []
    # [(string_to_datetime("2020-03-13 03:00:00.0"), string_to_datetime("2020-03-13 04:00:00.0"))]
    start_timestamp = string_to_datetime("2020-07-01 00:00:00.0")
    end_timestamp = string_to_datetime("2020-07-08 00:00:00.0")

    order_books = make_order_books(None, None)
    if order_books is None:
        agg_ticks = read_agg_ticks('C:/development/github/puffin-trader/tmp/',
                                   ['binance_agg_ticks.csv'],
                                   start_timestamp=start_timestamp,
                                   end_timestamp=end_timestamp,
                                   ignore_date_ranges=ignore_date_ranges)
        print(f'Agg ticks ({len(agg_ticks)}) {agg_ticks[0].timestamp} - {agg_ticks[-1].timestamp}')
        order_books = make_order_books(agg_ticks, timedelta(seconds=1))

    print(f'Order books ({len(order_books)}) {order_books[0].timestamp} - {order_books[-1].timestamp}')

    delta = 0.001
    runner = Runner(delta=delta, order_book=order_books[0])
    for order_book in order_books:
        runner.step(order_book)

    print(f'Runner: {len(runner.ie_times)}')

    prices = np.empty(len(order_books))
    asks = np.empty(len(order_books))
    bids = np.empty(len(order_books))
    times = np.empty(len(order_books))
    for idx, order_book in enumerate(order_books):
        prices[idx] = order_book.mid
        asks[idx] = order_book.ask
        bids[idx] = order_book.bid
        times[idx] = datetime.timestamp(order_book.timestamp)

    ax1 = plt.subplot(1, 1, 1)
    ax1.grid(True)
    plt.plot(times, prices, label=f'price')
    plt.plot(times, asks, label=f'ask')
    plt.plot(times, bids, label=f'bid')
    plt.plot(runner.os_times, runner.os_prices, label=f'OS')
    plt.scatter(runner.os_times, runner.os_prices, label=f'OS', s=5**2)
    plt.scatter(runner.dc_times, runner.dc_prices, label=f'DC', s=7**2)
    plt.scatter(runner.ie_times, runner.ie_prices, label=f'IE', s=5**2)

    plt.scatter(runner.ie_times, runner.ie_min_asks, label=f'Min Asks', s=5**2)
    plt.scatter(runner.ie_times, runner.ie_max_bids, label=f'Max Bids', s=5**2)

    plt.legend()

    plt.show()

    data = (delta,
            order_books,
            runner)

    with open(f"cache/intrinsic_time_data.pickle", 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    print(len(runner.ie_times))
