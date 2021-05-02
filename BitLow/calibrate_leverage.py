import os
import glob
import pickle
import pandas as pd


def get_prices():
    try:
        with open(f"cache/prices.pickle", 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        pass

    symbols = []
    for file_path in glob.glob("cache/klines/*.csv"):
        symbol = os.path.basename(file_path).replace('.csv', '')
        symbols.append(symbol)

    prices = {}
    for symbol in symbols:
        data = pd.read_csv(f"cache/klines/{symbol}.csv")
        symbol_prices = data['close'].to_numpy()
        prices[symbol] = symbol_prices

    with open(f"cache/prices.pickle", 'wb') as f:
        pickle.dump(prices, f, pickle.HIGHEST_PROTOCOL)

    return prices


def main():
    prices = get_prices()
    print(prices)


if __name__ == '__main__':
    main()
