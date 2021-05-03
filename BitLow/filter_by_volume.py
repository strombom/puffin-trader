import glob
import operator
import os
import pickle

import numpy as np
import pandas as pd


def calculate_volume():
    volumes = {}
    for file_path in glob.glob("cache/klines/*.csv"):
        symbol = os.path.basename(file_path).replace('.csv', '')
        print("Calculate volume", symbol)

        data = pd.read_csv(f"cache/klines/{symbol}.csv")

        symbol_prices = data['close'].to_numpy()
        symbol_volumes = data['volume'].to_numpy()
        volumes[symbol] = np.sum(symbol_prices * symbol_volumes)

    return volumes


def main():
    volumes = calculate_volume()

    print(volumes)
    volumes = sorted(volumes.items(), key=operator.itemgetter(1), reverse=True)

    print(volumes)


if __name__ == '__main__':
    main()
