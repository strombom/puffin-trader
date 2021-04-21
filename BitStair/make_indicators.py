import os
import glob
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
    # start_date = datetime.strptime('2021-01-01 00:00:00 UTC', '%Y-%m-%d %H:%M:%S %Z')
    start_date = datetime.strptime('2021-04-12 00:00:00 UTC', '%Y-%m-%d %H:%M:%S %Z')
    end_date = datetime.strptime('2021-04-17 00:00:00 UTC', '%Y-%m-%d %H:%M:%S %Z')
    delta = 0.005

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
        if count == 2:
            break

    import matplotlib.pyplot as plt

    indicators = None

    for pair_idx, pair in enumerate(pairs):
        data = pd.read_csv(f"cache/tickers/{pair}.csv")
        pair_start_date = datetime.utcfromtimestamp(data.iloc[0]['timestamp'] / 1000)
        pair_end_date = datetime.utcfromtimestamp(data.iloc[data.shape[0] - 1]['timestamp'] / 1000)
        if pair_start_date != start_date or pair_end_date < end_date:
            continue

        if indicators is None:
            n_pairs = len(pairs)
            n_lengths = len(lengths)
            n_timesteps = data.shape[0]
            indicators = np.empty((n_pairs, n_lengths, n_timesteps))

        dirs = []
        runner = Runner(delta=delta)

        runner_timestamps, runner_prices = [], []
        for idx, row in data.iterrows():
            for ie_event in runner.step(high=row['high'], low=row['low']):
                runner_timestamps.append(row['timestamp'])
                runner_prices.append(ie_event)
        runner_prices = np.array(runner_prices)

        # runner_prices = runner_prices / max(runner_prices)
        # plt.plot(timestamps, ie_events, label=f"{pair}")
        # print(pair, datetime.fromtimestamp(timestamps[200] / 1000).strftime('%Y-%m-%d %H:%M:%S'), runner_prices[200])

        directions = np.zeros((len(lengths), runner_prices.shape[0]))

        make_spectrum(lengths=lengths,
                      prices=runner_prices,
                      poly_order=direction_degree,
                      directions=directions)

        print(f"indicators.shape {indicators.shape}")
        print(f"directions.shape {directions.shape}")

        runner_idx = 0
        for kline_idx, row in data.iterrows():
            kline_timestamp = row['timestamp']

            while runner_idx < runner_prices.shape[0] - 1 and runner_timestamps[runner_idx + 1] <= kline_timestamp:
                runner_idx += 1

            # print(f"kline {kline_timestamp} runner {runner_timestamps[runner_idx]}")

            indicators[pair_idx, :, kline_idx] = directions[:, runner_idx]

        """
        fig, axs = plt.subplots(1 + 1, 1, sharex='all', gridspec_kw={'wspace': 0, 'hspace': 0})
        axs[0].plot(runner_prices[lengths[-1]:], label=f"{pair}")

        x = np.arange(runner_prices.shape[0] - lengths[-1])
        direction = directions[:, lengths[-1]:]

        # x = np.arange(runner_prices.shape[0])
        # direction = directions[:, :]

        direction_amplitude = np.max(np.abs(direction))
        direction = (direction_amplitude + direction) / (2 * direction_amplitude)
        axs[1].pcolormesh(x, lengths, direction,
                          vmin=np.min(direction), vmax=np.max(direction),
                          shading='auto', cmap=plt.get_cmap('RdYlGn'))
        axs[1].set_yscale('log')

        mouse_lines = MouseLines(fig=fig, axs=axs, min=min(runner_prices), max=max(runner_prices))

        plt.tight_layout()
        plt.legend()
        plt.show()
        quit()
        quit()
        """

        """
        break
        print(ie_events)
        print(timestamps)
        print(len(ie_events))
        print(data.shape)
        print(pair)
        quit()
        """

    print("a")

    # quit()
    # plt.legend()
    # plt.show()
