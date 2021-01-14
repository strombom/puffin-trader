import pickle
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open(f"cache/intrinsic_time_runner.pickle", 'rb') as f:
        delta, runner = pickle.load(f)

    # runner.os_prices = np.array(runner.os_prices)
    # runner.dc_prices = np.array(runner.dc_prices)
    runner.ie_prices = np.array(runner.ie_prices)
    x = np.arange(runner.ie_prices.shape[0])

    p = np.array(runner.ie_prices)
    ax1 = plt.subplot(1, 1, 1)
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
    plt.scatter(x, runner.ie_prices, label=f'IE', s=5 ** 2)

    # plt.scatter(runner.ie_times, runner.ie_min_asks, label=f'Min Asks', s=5 ** 2)
    # plt.scatter(runner.ie_times, runner.ie_max_bids, label=f'Max Bids', s=5 ** 2)

    plt.legend()

    plt.show()


