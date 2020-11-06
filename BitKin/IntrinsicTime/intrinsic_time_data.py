import math
import pickle
import numpy as np
from datetime import timedelta

from Common.OrderBook import make_order_books
from Common.misc import read_agg_ticks
from IntrinsicTime.runner import Runner, Direction


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

    target_direction = np.zeros((len(deltas), len(runner_clock.ie_times)))
    measured_direction = np.zeros((len(deltas), len(runner_clock.ie_times)))
    direction_changes = np.zeros((len(deltas), len(runner_clock.ie_times)))
    TMV = np.zeros((len(deltas), len(runner_clock.ie_times)))
    RET = np.zeros((len(deltas), len(runner_clock.ie_times)))

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

    # TMV
    for idx_runner, runner in enumerate(runners):
        idx_dc = 0
        ref_price = runner_clock.ie_prices[0]
        ref_time = runner_clock.ie_times[0]

        for idx_clock, timestamp in enumerate(runner_clock.ie_times):
            while idx_dc < len(runner.os_times) and runner.dc_times[idx_dc] < timestamp:
                idx_dc += 1
                ref_price = runner.os_prices[idx_dc - 1]
                ref_time = runner.os_times[idx_dc - 1]
            if idx_dc >= len(runner.os_times):
                break

            price = runner_clock.ie_prices[idx_clock]
            timestamp = runner_clock.ie_times[idx_clock]
            tmv = (price - ref_price) / (ref_price * deltas[idx_runner])
            if timestamp != ref_time:
                r = 1000 * tmv / (timestamp - ref_time)
                r = math.log(1 + abs(r)) * r / abs(r)
            else:
                r = 0

            if tmv != 0:
                tmv = math.log(1 + abs(tmv)) * tmv / abs(tmv)

            TMV[idx_runner, idx_clock] = tmv
            RET[idx_runner, idx_clock] = r


    feature_length = 100
    # (batches, feature_types, deltas, timesteps)

    features = np.zeros((len(runner_clock.ie_times) - feature_length, 3 * len(deltas) + 1, feature_length))
    targets = np.zeros((len(runner_clock.ie_times) - feature_length, len(deltas), feature_length))

    for x in range(len(runner_clock.ie_times) - feature_length):
        #md_feature = measured_direction[:, x:x + feature_length]
        #td_feature = target_direction[:, x:x + feature_length]
        #tmv_feature = TMV[:, x:x + feature_length]
        #ret_feature = RET[:, x:x + feature_length]

        targets[x] = target_direction[:, x:x + feature_length]
        #features[x, 0] = measured_direction[:, x:x + feature_length]

        confirmed_dc_feature = features[x, 0:len(deltas)]
        confirmed_dc_feature[:] = measured_direction[:, x:x + feature_length]
        td_feature = target_direction[:, x:x + feature_length]
        dc_feature = direction_changes[:, x:x + feature_length]
        for delta_idx in range(len(deltas)):
            idx = feature_length - 1
            while idx > -1 and dc_feature[delta_idx, idx] == 0:
                idx -= 1
            if idx >= 0:
                confirmed_dc_feature[delta_idx, 0: idx] = td_feature[delta_idx, 0: idx]

        features[x, len(deltas):2 * len(deltas)] = TMV[:, x:x + feature_length]
        features[x, 2 * len(deltas):3 * len(deltas)] = RET[:, x:x + feature_length]
        features[x, 3 * len(deltas)] = np.linspace(0, 1, feature_length)

    data = (deltas,
            order_books,
            runners,
            runner_clock,
            targets,
            features)

    with open(f"cache/intrinsic_time_data.pickle", 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

