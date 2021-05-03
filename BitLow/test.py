import os
import glob
import pandas as pd


def save_hdf():
    for file_path in glob.glob("cache/klines/*.hdf"):
        symbol = os.path.basename(file_path).replace('.hdf', '')
        print("Open", symbol)

        data = pd.read_hdf(f"cache/klines/{symbol}.hdf")

        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[col] = pd.to_numeric(data[col])

        data.to_hdf(
            f"cache/klines/{symbol}.hdf",
            key=symbol,
            mode='w',
            complevel=9,
            complib='blosc'
        )


def main():
    save_hdf()


if __name__ == '__main__':
    main()
