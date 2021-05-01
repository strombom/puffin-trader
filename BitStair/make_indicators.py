import os
import glob
import pickle

import numpy as np
import pandas as pd
from datetime import datetime

from IntrinsicTime.runner import Runner


class MouseLines(object):
    def __init__(self, fig, axs, min, max):
        self.fig = fig
        self.axs = axs
        self.figcanvas = self.fig.canvas
        self.range = [min, max]

        self.line1 = axs[0].plot([0, 0], self.range)[0]
        self.line2 = axs[1].plot([0, 0], [0, 50])[0]
        self.figcanvas.mpl_connect('motion_notify_event', self.mouse_move)

    def mouse_move(self, event):
        print(event.xdata)
        self.line1.set_data([event.xdata, event.xdata], self.range)
        self.line2.set_data([event.xdata, event.xdata], [0, 50])
        self.fig.canvas.draw()


def make_spectrum(lengths, prices, poly_order, directions):
    for length_idx, length in enumerate(lengths):
        vols = []
        for idx in range(lengths[-1], prices.shape[0]):
            start, end = idx - length, idx
            xp = np.arange(start, end)
            yp = np.poly1d(np.polyfit(xp, prices[start:end], poly_order))

            curve = yp(xp)
            direction = curve[-1] / curve[-2] - 1.0
            directions[length_idx, idx] = direction


if __name__ == '__main__':
    plot = False

    # start_date = datetime.strptime('2021-01-01 00:00:00 UTC', '%Y-%m-%d %H:%M:%S %Z')
    start_date = datetime.strptime('2021-03-01 00:00:00 UTC', '%Y-%m-%d %H:%M:%S %Z')
    end_date = datetime.strptime('2021-04-28 00:00:00 UTC', '%Y-%m-%d %H:%M:%S %Z')

    with open('cache/optim_deltas.pickle', 'rb') as f:
        optim_deltas = pickle.load(f)

    direction_degree = 3
    # lengths = np.arange(5, 50, 2)
    lengths = np.array([5, 6, 7, 8, 9, 10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82])
    lengths_df = pd.DataFrame(data={'length': lengths})
    lengths_df.to_csv('cache/regime_data_lengths.csv')

    pairs = []
    count = 0
    for file_path in glob.glob("cache/tickers/*.csv"):
        pair = os.path.basename(file_path).replace('.csv', '')
        pairs.append(pair)
        count += 1
        # if count == 30:
        #     break

    import matplotlib.pyplot as plt

    indicators = None

    if plot:
        fig, axs = plt.subplots(1 + count, 1, sharex='all', gridspec_kw={'wspace': 0, 'hspace': 0})

    prices = []

    for pair_idx, pair in enumerate(pairs):
        data = pd.read_csv(f"cache/tickers/{pair}.csv")
        pair_start_date = datetime.utcfromtimestamp(data.iloc[0]['timestamp'] / 1000)
        pair_end_date = datetime.utcfromtimestamp(data.iloc[data.shape[0] - 1]['timestamp'] / 1000)
        if pair_start_date != start_date or pair_end_date < end_date:
            continue

        pair_prices = data['close'].to_numpy()
        prices.append(np.expand_dims(np.copy(pair_prices), axis=0))
        if plot:
            pair_prices /= pair_prices.max()
            axs[0].plot(pair_prices, label=f"{pair}")

        if indicators is None:
            n_pairs = len(pairs)
            n_lengths = len(lengths)
            n_timesteps = data.shape[0]
            indicators = np.empty((n_pairs, n_lengths, n_timesteps))

        dirs = []
        runner = Runner(delta=optim_deltas[pair])

        runner_timestamps, runner_prices = [], []
        for idx, row in data.iterrows():
            for ie_event in runner.step(high=row['high'], low=row['low']):
                runner_timestamps.append(row['timestamp'])
                runner_prices.append(ie_event)
        runner_prices = np.array(runner_prices)

        directions = np.zeros((len(lengths), runner_prices.shape[0]))

        make_spectrum(lengths=lengths,
                      prices=runner_prices,
                      poly_order=direction_degree,
                      directions=directions)

        print(f"{pair} indicators {indicators.shape} directions {directions.shape}")

        runner_idx = 0
        for kline_idx, row in data.iterrows():
            kline_timestamp = row['timestamp']
            while runner_idx < runner_prices.shape[0] - 1 and runner_timestamps[runner_idx + 1] <= kline_timestamp:
                runner_idx += 1
            indicators[pair_idx, :, kline_idx] = directions[:, runner_idx]

        x = np.arange(indicators.shape[2])
        if plot:
            for length_idx in range(lengths.shape[0]):
                direction = indicators[pair_idx, :, :]
                direction_amplitude = 0.02
                # direction = (direction_amplitude + direction) / (2 * direction_amplitude)
                axs[1 + pair_idx].pcolormesh(
                    x, lengths, direction,
                    vmin=-direction_amplitude, vmax=direction_amplitude,
                    shading='auto', cmap=plt.get_cmap('RdYlGn')
                )

    if plot:
        axs[0].legend()
        plt.tight_layout()
        plt.show()

    prices = np.concatenate(prices, axis=0)

    with open(f"cache/indicators.pickle", 'wb') as f:
        pickle.dump({
            'pairs': pairs,
            'prices': prices,
            'lengths': lengths,
            'indicators': indicators
        }, f)
