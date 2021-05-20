import math
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import leastsq

from Common.Misc import string_to_datetime
import make_data


bitcoin_inception = string_to_datetime("2009-01-09 00:00:00.0")


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
        prices.append(price)
    return np.array(prices)


def rainbow_indicator_load_params():
    with open(f"cache/rainbow_params.pickle", 'rb') as f:
        params = pickle.load(f)
    return params


def rainbow_indicator(params, timestamp: datetime, price, rainbow_n=7):
    rainbow_top = 10 ** (params[0][0] * math.log((timestamp - bitcoin_inception).days) + params[0][1])
    rainbow_bot = 10 ** (params[1][0] * math.log((timestamp - bitcoin_inception).days) + params[1][1])
    log_top, log_bot = math.log(rainbow_top), math.log(rainbow_bot)
    step_size = (log_top - log_bot) / (rainbow_n - 2)
    log_top, log_bot = log_top + step_size, log_bot - step_size
    return (math.log(price) - log_bot) / (log_top - log_bot)


if __name__ == '__main__':
    #timestamps, prices = make_data.make_data()

    #if not os.path.exists('cache'):
    #    os.makedirs('cache')

    #with open(f"cache/bitstamp_hourly.pickle", 'wb') as f:
    #    pickle.dump((timestamps, prices), f, pickle.HIGHEST_PROTOCOL)

    timestamps_bitstamp, prices = make_data.get_data()
    timestamps_bitstamp, prices = np.array(timestamps_bitstamp), np.array(prices)
    print(f'start {timestamps_bitstamp[0]} - end {timestamps_bitstamp[-1]}')

    timestamps_extended = []
    timestamp = string_to_datetime("2011-01-01 00:00:00.0")
    while timestamp < string_to_datetime("2032-01-01 00:00:00.0"):
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

    timestamps_top = [string_to_datetime("2011-06-08 00:00:00.0"),
                      string_to_datetime("2013-11-30 00:00:00.0"),
                      string_to_datetime("2017-12-17 00:00:00.0")]
    prices_top = [32.0, 1150.0, 19500.0]
    rainbow_top_params = rainbow_regression(timestamps_top, prices_top)
    rainbow_top = rainbow_generate(timestamps_extended, rainbow_top_params)

    timestamps_bot = [string_to_datetime("2012-10-27 00:00:00.0"),
                      string_to_datetime("2016-05-22 00:00:00.0"),
                      string_to_datetime("2017-03-25 00:00:00.0"),
                      string_to_datetime("2020-10-08 00:00:00.0")]
    prices_bot = [9.5, 438.0, 900.0, 10557.0]
    regr_bot_params = rainbow_regression(timestamps_bot, prices_bot)
    rainbow_bot = rainbow_generate(timestamps_extended, regr_bot_params)

    with open(f"cache/rainbow_params.pickle", 'wb') as f:
        pickle.dump((rainbow_top_params, regr_bot_params), f, pickle.HIGHEST_PROTOCOL)

    print("Tops")
    for timestamp_find in timestamps_top:
        for timestamp, price in zip(timestamps_extended, rainbow_top):
            if timestamp > timestamp_find:
                print(" ", timestamp, price)
                break

    print("Bots")
    for timestamp_find in timestamps_bot:
        for timestamp, price in zip(timestamps_extended, rainbow_bot):
            if timestamp > timestamp_find:
                print(" ", timestamp, price)
                break

    rainbow_n = 9
    log_top, log_bot = np.log(rainbow_top), np.log(rainbow_bot)
    log_diff = (log_top - log_bot) / (rainbow_n - 2)
    log_bot = log_bot - log_diff
    log_diffs = np.arange(rainbow_n + 1)[..., np.newaxis] * log_diff
    rainbows = np.exp(log_diffs + log_bot)

    with open(f"cache/rainbow.pickle", 'wb') as f:
        pickle.dump((timestamps_extended, rainbows, timestamps_bitstamp, prices), f, pickle.HIGHEST_PROTOCOL)

    f, ax1 = plt.subplots(1, 1, sharex='all', gridspec_kw={'height_ratios': [1]})
    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.ticklabel_format(axis='y', style='plain')

    for n in range(rainbow_n):
        ax1.fill_between(timestamps_extended, rainbows[n], rainbows[n + 1],
                         facecolor=plt.get_cmap('gist_rainbow')((rainbow_n - 1 - n) / rainbow_n),
                         alpha=0.55)

    ax1.plot(timestamps_bitstamp, prices, c='white', linewidth=1.0)
    ax1.plot(timestamps_bitstamp, prices, c='black', linewidth=0.5, label=f'BTCUSD')
    ax1.legend()

    # ax2.grid(True)
    # ax1.set_yscale('lin')
    # ax2.plot(timestamps, diff, label=f'Diff')
    # ax2.fill_between(timestamps, 0, diff)
    # ax2.legend()

    plt.show()

# 10^(2.9065  * ln((number of days since 2009 Jan 09)/days) - 19.493)

# timestamps_2021 = timestamps[start_idx:end_idx]
# rainbow_2021 = rainbow_generate(timestamps_2021, [2.9065, 19.493])

# rainbow_2014 = []
# for timestamp in timestamps_2014:
#     days_since_inception = (timestamp - bitcoin_inception).days
#     r = 10 ** (2.9065 * math.log(days_since_inception) - 19.493)
#     rainbow_2014.append(r)

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
