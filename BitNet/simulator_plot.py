import pickle

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv(
        "log/simlog 2021-09-22 182620.txt",
        parse_dates=['date'],
        date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S%z")
    )

    """
    symbol = 'BTCUSDT'
    klines = pd.read_hdf(f"cache/klines/{symbol}.hdf")

    prices = []
    kline_idx = 525797
    for idx, row in klines.iterrows():
        while klines[symbol].iloc[kline_start_idx[symbol]]['open_time'] < start_timestamp_ms:
            kline_start_idx[symbol] += 1
    """

    fig, (ax1, ax2) = plt.subplots(2, sharex='col')
    ax1.plot(df['date'], df['equity'])
    ax1.set_yscale('log')
    ax2.plot(df['date'], 1 - df['cash'] / df['equity'])
    plt.show()


if __name__ == '__main__':
    main()
