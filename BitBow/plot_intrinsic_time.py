import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from Indicators.supersmoother import SuperSmoother
from rainbow import rainbow_indicator_load_params, rainbow_indicator


if __name__ == '__main__':
    with open(f"cache/intrinsic_time_runner.pickle", 'rb') as f:
        delta, runner = pickle.load(f)

    # runner.os_prices = np.array(runner.os_prices)
    # runner.dc_prices = np.array(runner.dc_prices)
    runner.ie_prices = np.array(runner.ie_prices)
    x = np.arange(runner.ie_prices.shape[0])

    rainbow_params = rainbow_indicator_load_params()
    rainbow = np.empty(x.shape[0])
    for idx, (timestamp, price) in enumerate(zip(runner.ie_times, runner.ie_prices)):
        timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        rainbow[idx] = rainbow_indicator(params=rainbow_params, timestamp=timestamp, price=price)

    smooth_periods = [200, 2000]
    smooths = {}
    for smooth_period in smooth_periods:
        smooth = []
        smoother = SuperSmoother(period=smooth_period, initial_value=runner.ie_prices[0])
        for price in runner.ie_prices:
            smooth.append(smoother.append(price))
        smooths[smooth_period] = smooth

    p = np.array(runner.ie_prices)
    f, (ax1, ax2) = plt.subplots(2, 1, sharex='all', gridspec_kw={'height_ratios': [3, 1]})
    # ax1 = plt.subplot(2, 1, 1)
    ax1.grid(True)
    ax1.set_yscale('log')
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

    # ax2 = plt.subplot(2, 1, 2)
    ax2.set_ylim((0, 1))
    ax2.plot(x, rainbow, label=f'Rainbow')



    # plt.scatter(runner.ie_times, runner.ie_min_asks, label=f'Min Asks', s=5 ** 2)
    # plt.scatter(runner.ie_times, runner.ie_max_bids, label=f'Max Bids', s=5 ** 2)

    plt.legend()

    plt.show()


