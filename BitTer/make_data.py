
import pickle
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from Common.AggTicks import read_agg_ticks
from Common.ESeries import make_e_series_range
from Common.Misc import string_to_datetime
from Common.OrderBook import make_order_books
from IntrinsicTime.runner import Runner, Direction


if __name__ == "__main__":

    ignore_date_ranges = []
    # [(string_to_datetime("2020-03-13 03:00:00.0"), string_to_datetime("2020-03-13 04:00:00.0"))]
    start_timestamp = string_to_datetime("2020-01-01 00:00:00.0")
    end_timestamp = string_to_datetime("2020-02-01 00:00:00.0")

    order_books = make_order_books(None, None)
    if order_books is None:
        agg_ticks = read_agg_ticks('C:/development/github/puffin-trader/tmp/',
                                   ['agg_ticks_1.csv', 'agg_ticks_2.csv', 'agg_ticks_3.csv'],
                                   start_timestamp=start_timestamp,
                                   end_timestamp=end_timestamp,
                                   ignore_date_ranges=ignore_date_ranges)
        print(f'Agg ticks ({len(agg_ticks)}) {agg_ticks[0].timestamp} - {agg_ticks[-1].timestamp}')
        order_books = make_order_books(agg_ticks, timedelta(seconds=1))

    print(f'Order books ({len(order_books)}) {order_books[0].timestamp} - {order_books[-1].timestamp}')

    # delta_clock = 0.0022
    # deltas = make_e_series_range(delta_clock, 0.02, 96)
    delta_clock = 0.0022
    #deltas = [delta_clock]
    deltas = make_e_series_range(delta_clock, 0.04, 48)

    print(f'Deltas({len(deltas)}): {deltas}')

    runners = []
    for delta in deltas:
        runners.append(Runner(delta=delta, order_book=order_books[0]))
    runner_clock = Runner(delta=delta_clock, order_book=order_books[0])

    for order_book in order_books:
        runner_clock.step(order_book)
        for runner in runners:
            event = runner.step(order_book)
    print("Runners done")
    print(f'Runner({deltas[0]}): {len(runners[0].ie_times)}, runner({deltas[-1]}): {len(runners[-1].ie_times)}')

    target_direction = np.zeros((len(deltas), len(runner_clock.ie_times)))
    measured_direction = np.zeros((len(deltas), len(runner_clock.ie_times)))
    direction_changes = np.zeros((len(deltas), len(runner_clock.ie_times)))
    clock_TMV = np.zeros((len(deltas), len(runner_clock.ie_times)))
    clock_R = np.zeros((len(deltas), len(runner_clock.ie_times)))

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
                target_direction[idx_runner, idx_clock] = 1
            else:
                target_direction[idx_runner, idx_clock] = 0
    print("Target directions done")

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
                measured_direction[idx_runner, idx_clock] = 1
            else:
                measured_direction[idx_runner, idx_clock] = 0
    print("Measured directions done")

    # Direction changes
    for idx_runner, runner in enumerate(runners):
        idx_dc = 0
        for idx_clock, timestamp in enumerate(runner_clock.ie_times):
            found = False
            while idx_dc < len(runner.dc_times) and runner.dc_times[idx_dc] < timestamp:
                idx_dc += 1
                found = True

            if idx_dc >= len(runner.dc_times):
                break

            if found:
                direction_changes[idx_runner, idx_clock] = 1
    print("Direction changes done")

    # IE clocked TMV, R
    for idx_runner, runner in enumerate(runners):
        idx_dc = 0
        ref_price = runner_clock.ie_prices[0]
        ref_time = runner_clock.ie_times[0]

        for idx_clock, timestamp in enumerate(runner_clock.ie_times):
            price = runner_clock.ie_prices[idx_clock]
            timestamp = runner_clock.ie_times[idx_clock]

            while idx_dc + 1 < len(runner.os_times) and runner.dc_times[idx_dc + 1] <= timestamp:
                idx_dc += 1
                ref_price = runner.os_prices[idx_dc]
                ref_time = runner.os_times[idx_dc]
            #if idx_dc >= len(runner.os_times) - 1:
            #    break

            tmv = (price - ref_price) / (ref_price * deltas[idx_runner])

            if timestamp != ref_time:
                r = 1000 * tmv / (timestamp - ref_time)
            else:
                r = 0

            clock_TMV[idx_runner, idx_clock] = tmv
            clock_R[idx_runner, idx_clock] = r

    r_pos = np.log1p(clock_R.clip(min=0))
    r_neg = np.log1p(-clock_R.clip(max=0))
    for idx in range(len(deltas)):
        r_k = 0.5 / (0.1344193 * deltas[idx] ** -0.54606854 - 0.89771839 + 0.14)
        clock_R[idx] = r_k * (r_pos[idx] - r_neg[idx])

    clock_TMV = clock_TMV / 5

    print("TMV, R done")

    # prices = np.empty(len(order_books))
    # asks = np.empty(len(order_books))
    # bids = np.empty(len(order_books))
    # times = np.empty(len(order_books))
    # for idx, order_book in enumerate(order_books):
    #     prices[idx] = order_book.mid
    #     asks[idx] = order_book.ask
    #     bids[idx] = order_book.bid
    #     times[idx] = datetime.timestamp(order_book.timestamp)
    #
    # plt.plot(times, prices, label=f'price')
    # plt.plot(times, asks, label=f'ask')
    # plt.plot(times, bids, label=f'bid')
    #
    # for i in [0, 3]:
    #     plt.plot(runners[i].os_times, runners[i].os_prices, label=f'OS {i}')
    #     plt.scatter(runners[i].os_times, runners[i].os_prices, label=f'OS {i}', s=10**2)
    #     plt.scatter(runners[i].dc_times, runners[i].dc_prices, label=f'DC {i}', s=8**2)
    #     plt.scatter(runners[i].ie_times, runners[i].ie_prices, label=f'IE {i}', s=6**2)
    #
    # plt.legend()
    # plt.show()

    #quit()

    data = (deltas,
            order_books,
            runners,
            runner_clock,
            clock_TMV,
            clock_R)

    with open(f"cache/intrinsic_time_data.pickle", 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    print(len(deltas), deltas)
    print(len(runner_clock.ie_times))
    print(len(runners[-1].ie_times))

    quit()

    filename = 'C:/development/github/puffin-trader/BitKin/IntrinsicTime/cache/intrinsic_time_data.pickle'

    with open(filename, 'rb') as f:
        deltas, order_books, runners, runner_clock = pickle.load(f)

    print(len(deltas), deltas)
    print(len(runner_clock.ie_times))
    print(len(runners[-1].ie_times))

    features = np.empty((len(runner_clock.ie_times), 2 * len(runners)))

    # features = np.empty((len(runner_clock.ie_times) - feature_length, len(deltas), feature_length))
