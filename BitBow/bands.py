import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

from Common.Misc import string_to_datetime
from make_data import get_data


bitcoin_inception = string_to_datetime("2009-01-09 00:00:00.0")


""""
def rainbow_prices(timestamp):
    days_since_inception = (timestamp - bitcoin_inception).days
    rainbow = 10 ** (2.9065 * math.log(days_since_inception) - 19.493)
    print(rainbow)
"""


def rainbow_regression_f(x, prices, timestamps):
    y = []
    for timestamp in timestamps:
        price = 10 ** (x[0] * math.log((timestamp - bitcoin_inception).days) + x[1])
        y.append((price))
    return np.log(prices) - np.log(y)


def rainbow_regression(timestamps, prices):
    #days_since_inception = (first_timestamp - bitcoin_inception).days
    prices = (np.array(prices))
    x0 = np.ones(2)
    result = leastsq(func=rainbow_regression_f, x0=x0, args=(prices, timestamps))
    print(result)
    return result[0]


def rainbow_generate(timestamps, params):
    days_since_inception = (timestamps[0] - bitcoin_inception).days
    prices = []
    for timestamp in timestamps:
        price = 10 ** (params[0] * math.log((timestamp - bitcoin_inception).days) + params[1])
        prices.append((price))
        #price = np.power(10, params[0] * np.log(np.arange(timestamps.shape[0]) / 24 + days_since_inception) + params[1])
    return np.array(prices)


if __name__ == '__main__':
    # print(rainbow_prices(string_to_datetime("2020-01-01 00:00:00.0")))
    # quit()

    timestamps, prices = get_data()
    timestamps, prices = np.array(timestamps), np.array(prices)
    print(f'start {timestamps[0]} - end {timestamps[-1]}')

    start_idx = 0
    for idx, timestamp in enumerate(timestamps):
        if timestamp >= string_to_datetime("2012-01-01 00:00:00.0"):
            start_idx = idx
            break

    end_idx = 0
    for idx, timestamp in enumerate(timestamps):
        if timestamp >= string_to_datetime("2019-06-01 00:00:00.0"):
            end_idx = idx
            break

    print(f'start {timestamps[start_idx]} - end {timestamps[end_idx]}')

    # timestamps_2021 = timestamps[start_idx:end_idx]
    # rainbow_2021 = rainbow_generate(timestamps_2021, [2.9065, 19.493])

    # rainbow_2014 = []
    # for timestamp in timestamps_2014:
    #     days_since_inception = (timestamp - bitcoin_inception).days
    #     r = 10 ** (2.9065 * math.log(days_since_inception) - 19.493)
    #     rainbow_2014.append(r)

    timestamps_bottom = [string_to_datetime("2011-11-20 00:00:00.0"),
                         string_to_datetime("2012-05-01 00:00:00.0"),
                         string_to_datetime("2013-01-01 00:00:00.0"),
                         string_to_datetime("2015-09-01 00:00:00.0"),
                         string_to_datetime("2016-10-01 00:00:00.0"),
                         string_to_datetime("2017-03-01 00:00:00.0"),
                         string_to_datetime("2019-02-01 00:00:00.0"),
                         string_to_datetime("2020-07-01 00:00:00.0"),
                         string_to_datetime("2020-10-01 00:00:00.0")]
    prices_bottom = [2.0, 5.0, 13.0, 226.0, 609.0, 900.0, 3400.0, 9150.0, 10550.0]
    rainbow_bottom_params = rainbow_regression(timestamps_bottom, prices_bottom)
    rainbow_bottom = rainbow_generate(timestamps, rainbow_bottom_params)

    timestamps_top = [string_to_datetime("2011-11-20 00:00:00.0"),
                      string_to_datetime("2013-11-30 00:00:00.0"),
                      string_to_datetime("2017-12-17 00:00:00.0")]
    prices_top = [32.0, 1158.0, 19500.0]
    rainbow_top_params = rainbow_regression(timestamps_top, prices_top)
    rainbow_top = rainbow_generate(timestamps, rainbow_top_params)

    o_timestamps_top = [string_to_datetime("2012-11-21 00:00:00.0"),
                      string_to_datetime("2016-07-10 00:00:00.0"),
                      string_to_datetime("2020-05-09 00:00:00.0")]
    o_prices_top = [36.0, 1830.0, 23350.0]
    o_rainbow_top_params = rainbow_regression(o_timestamps_top, o_prices_top)
    o_rainbow_top = rainbow_generate(timestamps, o_rainbow_top_params)

    o_timestamps_mid = [string_to_datetime("2012-11-21 00:00:00.0"),
                      string_to_datetime("2016-07-10 00:00:00.0"),
                      string_to_datetime("2020-05-09 00:00:00.0")]
    o_prices_mid = [36.0, 1830.0, 23350.0]
    o_rainbow_mid_params = rainbow_regression(o_timestamps_mid, o_prices_mid)
    o_rainbow_mid = rainbow_generate(timestamps, o_rainbow_mid_params)

    o_timestamps_bot = [string_to_datetime("2012-11-21 00:00:00.0"),
                      string_to_datetime("2016-07-10 00:00:00.0"),
                      string_to_datetime("2020-05-09 00:00:00.0")]
    o_prices_bot = [36.0, 1830.0, 23350.0]
    o_rainbow_bot_params = rainbow_regression(o_timestamps_bot, o_prices_bot)
    o_rainbow_bot = rainbow_generate(timestamps, o_rainbow_bot_params)

    rainbow_params = rainbow_regression(timestamps[start_idx:end_idx], prices[start_idx:end_idx])
    timestamps_2021 = timestamps
    rainbow_2021 = rainbow_generate(timestamps_2021, rainbow_params)

    timestamps_short = timestamps[start_idx:end_idx]
    rainbow_short = rainbow_generate(timestamps_short, rainbow_params)

    ax1 = plt.subplot(1, 1, 1)
    ax1.grid(True)
    plt.yscale('log')
    plt.plot(timestamps, prices, label=f'Price')
    # plt.plot(timestamps_2014, rainbow_2014, label=f'Rainbow 2014')
    plt.plot(timestamps_2021, rainbow_2021, label=f'Rainbow 2021')
    plt.plot(timestamps_short, rainbow_short, label=f'Rainbow short')
    plt.plot(timestamps, rainbow_bottom, label=f'Rainbow bottom')
    plt.plot(timestamps, rainbow_top, label=f'Rainbow top')
    plt.plot(timestamps, np.exp((np.log(rainbow_bottom) + np.log(rainbow_top)) / 2), label=f'Rainbow mid')
    plt.plot(timestamps, o_rainbow_top, label=f'Orig Rainbow top')
    plt.plot(timestamps, o_rainbow_mid, label=f'Orig Rainbow mid')
    plt.plot(timestamps, o_rainbow_bot, label=f'Orig Rainbow bot')
    plt.scatter(timestamps_bottom, prices_bottom, label=f'Rainbow bottom', c='red')
    plt.legend()
    plt.show()

# 10^(2.9065  * ln((number of days since 2009 Jan 09)/days) - 19.493)
