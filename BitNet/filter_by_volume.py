import os
import glob
import pickle
import operator
import numpy as np
import pandas as pd
from datetime import datetime


def calculate_volume():
    start_timestamp = datetime.strptime("2020-01-01 01:00:00", "%Y-%m-%d %H:%M:%S")
    min_volatility = 0.1
    min_volume = 20_000_000_000

    volatilities = {}
    volumes = {}
    symbols = []

    for file_path in glob.glob("cache/klines/*.hdf"):
        symbol = os.path.basename(file_path).replace('.hdf', '')

        if 'USDT' not in symbol:
            continue

        data = pd.read_hdf(file_path)
        timestamp = datetime.fromtimestamp(data['open_time'].iloc[0] / 1000, tz=timezone.utc)

        if timestamp != start_timestamp:
            continue

        volatility = data['close'].std() / data['close'].mean()
        volume = np.sum(data['close'].to_numpy() * data['volume'].to_numpy())

        if volatility < min_volatility or volume < min_volume:
            continue

        volatilities[symbol] = volatility
        volumes[symbol] = volume
        symbols.append(symbol)

        print(timestamp, symbol, volatility)

    volumes = sorted(volumes.items(), key=operator.itemgetter(1), reverse=True)
    volatilities = sorted(volatilities.items(), key=operator.itemgetter(1), reverse=True)
    print(len(volumes), volumes)
    print(len(volatilities), volatilities)

    return symbols


def main():
    symbols = calculate_volume()

    with open(f"cache/filtered_symbols.pickle", 'wb') as f:
        pickle.dump(symbols, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
