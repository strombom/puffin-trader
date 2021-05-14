import glob
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from IntrinsicTime.runner import Runner


def make_spectrum(lengths, prices, poly_order, directions):
    for length_idx, length in enumerate(lengths):
        for idx in range(lengths[-1], prices.shape[0]):
            start, end = idx - length, idx
            xp = np.arange(start, end)
            yp = np.poly1d(np.polyfit(xp, prices[start:end], poly_order))

            curve = yp(xp)
            direction = curve[-1] / curve[-2] - 1.0
            directions[length_idx, idx] = direction


def main():
    with open(f"cache/filtered_symbols.pickle", 'rb') as f:
        symbols = pickle.load(f)

    with open(f"cache/optim_deltas.pickle", 'rb') as f:
        optim_deltas = pickle.load(f)

    start_timestamp = datetime.strptime("2020-02-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    # end_timestamp = datetime.strptime("2020-01-24 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_timestamp = datetime.strptime("2021-05-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    klines_count = int((end_timestamp - start_timestamp).total_seconds() / 60)

    direction_degree = 2
    lengths = np.array([10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82])
    lengths_df = pd.DataFrame(data={'length': lengths})
    lengths_df.to_csv('cache/regime_data_lengths.csv')

    n_timesteps = int((end_timestamp - start_timestamp).total_seconds() / 60)
    indicators = np.empty((len(symbols), len(lengths), n_timesteps))

    prices = []

    for symbol_idx, symbol in enumerate(symbols):
        klines = pd.read_hdf(f"cache/klines/{symbol}.hdf")

        symbol_prices = []
        runner = Runner(delta=1.0)
        runner_timestamps, runner_prices = [], []
        for row_idx, row in klines.iterrows():
            timestamp = datetime.fromtimestamp(row['open_time'] / 1000, tz=timezone.utc)
            if timestamp < start_timestamp:
                continue
            if timestamp >= end_timestamp:
                break

            symbol_prices.append(row['close'])

            timestamp = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            runner.delta = optim_deltas[timestamp][symbol]

            for ie_event in runner.step(price=row['close']):
                runner_timestamps.append(row['open_time'])
                runner_prices.append(ie_event)
        runner_prices = np.array(runner_prices)

        directions = np.zeros((len(lengths), runner_prices.shape[0]))

        make_spectrum(lengths=lengths,
                      prices=runner_prices,
                      poly_order=direction_degree,
                      directions=directions)

        symbol_prices = np.array(symbol_prices)
        prices.append(np.expand_dims(symbol_prices, axis=0))

        runner_idx = 0
        kline_idx = 0
        for _, row in klines.iterrows():
            kline_timestamp = datetime.fromtimestamp(row['open_time'] / 1000, tz=timezone.utc)
            if kline_timestamp < start_timestamp:
                continue
            if kline_timestamp >= end_timestamp:
                break

            while runner_idx < runner_prices.shape[0] - 1 and runner_timestamps[runner_idx + 1] <= kline_timestamp.timestamp() * 1000:
                runner_idx += 1
            indicators[symbol_idx, :, kline_idx] = directions[:, runner_idx]
            kline_idx += 1

    prices = np.concatenate(prices, axis=0)

    with open(f"cache/indicators.pickle", 'wb') as f:
        pickle.dump({
            'symbols': symbols,
            'prices': prices,
            'lengths': lengths,
            'indicators': indicators
        }, f)


if __name__ == '__main__':
    main()