
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import make_data
from Common.Misc import string_to_datetime


def main():
    timestamps_bitstamp, prices = make_data.get_data()
    timestamps_bitstamp, prices = np.array(timestamps_bitstamp), np.array(prices)
    print(f'start {timestamps_bitstamp[0]} - end {timestamps_bitstamp[-1]}')


    drawdowns = []
    max_price = 0
    for price in prices:
        max_price = max(price, max_price)
        drawdowns.append(price / max_price)

    f, axs = plt.subplots(2, 1, sharex='all', gridspec_kw={'height_ratios': [5, 2]})
    ax1, ax2 = axs

    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.ticklabel_format(axis='y', style='plain')

    # ax1.plot(timestamps_bitstamp, prices, c='white', linewidth=1.0)
    ax1.plot(timestamps_bitstamp, prices, c='black', linewidth=0.5, label=f'BTCUSD')
    ax1.legend()

    ax2.grid(True)
    # ax1.set_yscale('lin')
    ax2.plot(timestamps_bitstamp, drawdowns, linewidth=0.5, label=f'Drawdown')
    ax2.legend()

    plt.show()




if __name__ == '__main__':
    main()
