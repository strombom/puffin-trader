import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
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
        y.append(price)
    return np.log(prices) - np.log(y)


def rainbow_regression(timestamps, prices):
    # days_since_inception = (first_timestamp - bitcoin_inception).days
    prices = (np.array(prices))
    x0 = np.ones(2)
    result = leastsq(func=rainbow_regression_f, x0=x0, args=(prices, timestamps))
    # print(result)
    return result[0]


def rainbow_generate(timestamps, params):
    days_since_inception = (timestamps[0] - bitcoin_inception).days
    prices = []
    for timestamp in timestamps:
        price = 10 ** (params[0] * math.log((timestamp - bitcoin_inception).days) + params[1])
        prices.append((price))
        # price = np.power(10, params[0] * np.log(np.arange(timestamps.shape[0]) / 24 + days_since_inception) + params[1])
    return np.array(prices)


if __name__ == '__main__':
    # print(rainbow_prices(string_to_datetime("2020-01-01 00:00:00.0")))
    # quit()

    timestamps_bitstamp, prices = get_data()
    timestamps_bitstamp, prices = np.array(timestamps_bitstamp), np.array(prices)
    print(f'start {timestamps_bitstamp[0]} - end {timestamps_bitstamp[-1]}')

    timestamps_extended = []
    timestamp = string_to_datetime("2011-01-01 00:00:00.0")
    while timestamp < string_to_datetime("2022-01-01 00:00:00.0"):
        timestamps_extended.append(timestamp)
        timestamp += timedelta(hours=1)

    start_idx = 0
    for idx, timestamp in enumerate(timestamps_bitstamp):
        if timestamp >= string_to_datetime("2012-06-01 00:00:00.0"):
            start_idx = idx
            break

    end_idx = 0
    for idx, timestamp in enumerate(timestamps_bitstamp):
        if timestamp >= string_to_datetime("2019-09-01 00:00:00.0"):
            end_idx = idx
            break

    print(f'start {timestamps_bitstamp[start_idx]} - end {timestamps_bitstamp[end_idx]}')

    # timestamps_2021 = timestamps[start_idx:end_idx]
    # rainbow_2021 = rainbow_generate(timestamps_2021, [2.9065, 19.493])

    # rainbow_2014 = []
    # for timestamp in timestamps_2014:
    #     days_since_inception = (timestamp - bitcoin_inception).days
    #     r = 10 ** (2.9065 * math.log(days_since_inception) - 19.493)
    #     rainbow_2014.append(r)

    timestamps_top = [string_to_datetime("2011-06-08 00:00:00.0"),
                      string_to_datetime("2013-11-30 00:00:00.0"),
                      string_to_datetime("2017-12-17 00:00:00.0")]
    prices_top = [32.0, 1150.0, 19500.0]
    rainbow_top_params = rainbow_regression(timestamps_top, prices_top)
    rainbow_top = rainbow_generate(timestamps_extended, rainbow_top_params)

    for timestamp_find in timestamps_top:
        for timestamp, price in zip(timestamps_extended, rainbow_top):
            if timestamp > timestamp_find:
                print(timestamp, price)
                break

    timestamps_bottom = [string_to_datetime("2012-10-27 00:00:00.0"),
                         string_to_datetime("2015-08-24 00:00:00.0"),
                         string_to_datetime("2020-03-23 00:00:00.0")]
    prices_bottom = [9.5, 202.0, 5816.0]
    regr_bottom_params = rainbow_regression(timestamps_bottom, prices_bottom)
    regr_bottom = rainbow_generate(timestamps_extended, regr_bottom_params)

    # rainbow_params = rainbow_regression(timestamps_bitstamp[start_idx:end_idx], prices[start_idx:end_idx])
    # rainbow_2021 = rainbow_generate(timestamps_extended, rainbow_params)
    #
    # timestamps_short = timestamps_bitstamp[start_idx:end_idx]
    # rainbow_short = rainbow_generate(timestamps_short, rainbow_params)
    #
    # factors = []
    # for timestamp in timestamps_extended:
    #     # price = 10 ** (params[0] * math.log((timestamp - bitcoin_inception).days) + params[1])
    #     days = (timestamp - bitcoin_inception).days
    #     factors.append(math.pow(0.5, (days - 850) / (4 * 365)) * 20)
    # rainbow_top = rainbow_2021 * factors
    #
    # factors = []
    # for timestamp in timestamps_extended:
    #     # price = 10 ** (params[0] * math.log((timestamp - bitcoin_inception).days) + params[1])
    #     days = (timestamp - bitcoin_inception).days
    #     factors.append(38.1 * days ** -0.3727)
    # rainbow_bot = rainbow_2021 / factors
    # # print(np.array(factors))

    """
    timestamps_next = [string_to_datetime("2011-06-07 00:00:00.0"),
                       string_to_datetime("2013-12-04 00:00:00.0"),
                       string_to_datetime("2017-12-17 00:00:00.0"),
                       string_to_datetime("2021-12-17 00:00:00.0"),
                       string_to_datetime("2025-12-17 00:00:00.0"),
                       string_to_datetime("2029-12-17 00:00:00.0"),
                       string_to_datetime("2034-12-17 00:00:00.0")]
    rainbow_next = rainbow_generate(timestamps_next, rainbow_bottom_params)
    for price in rainbow_next * np.array([48, 24, 12, 6, 3, 1.5, 0.75]): # * 8.2:
        print(price)
    quit()
    """

    """
    timestamps = [string_to_datetime("2011-06-07 00:00:00.0"),
                  string_to_datetime("2013-12-04 00:00:00.0"),
                  string_to_datetime("2017-12-17 00:00:00.0")]
    rainbow_bottom = rainbow_generate(timestamps, rainbow_bottom_params)
    print(rainbow_bottom)
    quit()
    """

    """
    o_timestamps_top = [string_to_datetime("2012-11-14 00:00:00.0"),
                      string_to_datetime("2016-07-11 00:00:00.0"),
                      string_to_datetime("2020-05-11 00:00:00.0")]
    o_prices_top = [138.0, 6400.0, 79500.0]
    o_rainbow_top_params = rainbow_regression(o_timestamps_top, o_prices_top)
    o_rainbow_top = rainbow_generate(timestamps, o_rainbow_top_params)

    o_timestamps_mid = [string_to_datetime("2012-11-21 00:00:00.0"),
                      string_to_datetime("2016-07-10 00:00:00.0"),
                      string_to_datetime("2020-05-09 00:00:00.0")]
    o_prices_mid = [36.0, 1830.0, 23350.0]
    o_rainbow_mid_params = rainbow_regression(o_timestamps_mid, o_prices_mid)
    o_rainbow_mid = rainbow_generate(timestamps, o_rainbow_mid_params)

    o_timestamps_bot = [string_to_datetime("2012-11-22 00:00:00.0"),
                      string_to_datetime("2016-07-08 00:00:00.0"),
                      string_to_datetime("2020-05-02 00:00:00.0")]
    o_prices_bot = [9.4, 520.0, 6700.0]
    o_rainbow_bot_params = rainbow_regression(o_timestamps_bot, o_prices_bot)
    o_rainbow_bot = rainbow_generate(timestamps, o_rainbow_bot_params)
    """

    # diff = np.log(prices) - np.log(rainbow_2021)
    # print(diff)
    # quit()

    #bot2 = rainbow_2021 * 0.5

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # ax1 = plt.subplot(2, 1, 1)
    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.plot(timestamps_bitstamp, prices, label=f'Price')
    # plt.plot(timestamps_2014, rainbow_2014, label=f'Rainbow 2014')
    # ax1.plot(timestamps_extended, rainbow_2021, label=f'Rainbow 2021')
    # ax1.plot(timestamps_short, rainbow_short, label=f'Rainbow short')
    ax1.plot(timestamps_extended, regr_bottom, label=f'Regr bottom')
    ax1.plot(timestamps_extended, rainbow_top, label=f'Rainbow top')
    # ax1.plot(timestamps_2021, rainbow_bot, label=f'Rainbow bot')
    # ax1.plot(timestamps, np.exp((np.log(rainbow_bottom) + np.log(rainbow_top)) / 2), label=f'Rainbow mid')
    # plt.plot(timestamps, o_rainbow_top, label=f'Orig Rainbow top')
    # plt.plot(timestamps, o_rainbow_mid, label=f'Orig Rainbow mid')
    # plt.plot(timestamps, o_rainbow_bot, label=f'Orig Rainbow bot')
    ax1.scatter(timestamps_bottom, prices_bottom, label=f'Rainbow bottom', c='red')
    ax1.legend()

    # ax2 = plt.subplot(2, 1, 2)
    ax2.grid(True)
    # ax1.set_yscale('lin')
    # ax2.plot(timestamps, diff, label=f'Diff')
    # ax2.fill_between(timestamps, 0, diff)
    ax2.legend()

    plt.show()

# 10^(2.9065  * ln((number of days since 2009 Jan 09)/days) - 19.493)
