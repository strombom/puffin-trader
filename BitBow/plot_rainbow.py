import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator

from BitmexSim.bitmex_simulator import BitmexSimulator
from Common.Misc import string_to_datetime
from rainbow import rainbow_indicator_load_params, rainbow_indicator

if __name__ == '__main__':
    with open(f"cache/rainbow.pickle", 'rb') as f:
        timestamps_extended, rainbows, timestamps_bitstamp, prices_bitstamp = pickle.load(f)

    print(f'start {timestamps_bitstamp[0]} - end {timestamps_bitstamp[-1]}')
    rainbow_params = rainbow_indicator_load_params()

    # a = rainbow_indicator(rainbow_params, string_to_datetime("2021-01-13 00:00:00.0"), 35000.0)
    rainbow_indicators = []
    for timestamp, price in zip(timestamps_bitstamp, prices_bitstamp):
        rainbow_indicators.append(rainbow_indicator(rainbow_params, timestamp, price))

    # BitmexSimulator

    f, (ax1, ax2) = plt.subplots(2, 1, sharex='all', gridspec_kw={'height_ratios': [3, 1]})
    ax1.grid(which='minor', alpha=0.15)
    ax1.grid(which='major', alpha=0.4)
    ax1.set_yscale('log')
    ax1.set_xscale('linear') # , base=1)
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.ticklabel_format(axis='y', style='plain')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())

    for n in range(rainbows.shape[0] - 1):
        ax1.fill_between(timestamps_extended, rainbows[n], rainbows[n + 1],
                         facecolor=plt.get_cmap('gist_rainbow')((rainbows.shape[0] - 2 - n) / rainbows.shape[0]),
                         alpha=0.55)

    ax1.plot(timestamps_bitstamp, prices_bitstamp, c='white', linewidth=1.0)
    ax1.plot(timestamps_bitstamp, prices_bitstamp, c='black', linewidth=0.5, label=f'Bitcoin price')
    ax1.legend(loc='upper left')

    ax2.grid(which='minor', alpha=0.15)
    ax2.grid(which='major', alpha=0.4)
    # ax2.set_yscale('lin')
    ax2.plot(timestamps_bitstamp, rainbow_indicators, label=f'Rainbow indicator')
    # ax2.plot(timestamps, diff, label=f'Diff')
    # ax2.fill_between(timestamps, 0, diff)
    ax2.set_ylim([0, 1])
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
