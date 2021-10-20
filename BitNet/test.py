"""
import glob
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from binance_account import BinanceAccount
"""
import glob

import pandas as pd
import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt


def print_stats(v):

    mean = torch.mean(v)
    diffs = v - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0

    print(f"mean: {mean}")
    print(f"var: {var}")
    print(f"std: {std}")
    print(f"zscores: {zscores}")
    print(f"skews: {skews}")
    print(f"kurtoses: {kurtoses}")


def main():

    ups, downs = 0, 0
    for file_path in glob.glob("E:/BitBot/training_data/*.csv"):
        df = pd.read_csv(file_path)
        start_idx = int(2/3 * df.shape[0])
        ground_truth = df[start_idx:]['(1.05,0.95)'].to_numpy()
        print(file_path, ground_truth.mean())
        ups += (ground_truth > 0.0).nonzero()[0].shape[0]
        downs += (ground_truth < 0.0).nonzero()[0].shape[0]

    print(f"{ups}, {downs}, {ups / (ups + downs): .2f}")

    quit()

    def print_up_down(p):
        ups = (p > 0.0).nonzero().squeeze()
        downs = (p < 0.0).nonzero().squeeze()

        if len(downs.shape) == 0:
            down_count = 0
        else:
            down_count = downs.shape[0]

        if len(ups.shape) == 0:
            up_count = 0
        else:
            up_count = ups.shape[0]

        if down_count + up_count == 0:
            print(f"  {up_count} up, {down_count} down, tot {up_count + down_count}, ratio -", end='')
        else:
            print(f"  {up_count} up, {down_count} down, tot {up_count + down_count}, ratio {up_count / (up_count + down_count):.2f}", end='')

    with open('preds_2020-03-22.pickle', 'rb') as f:
        predictions = pickle.load(f)

    pred_train_mean = predictions['pred_train'].mean()
    gt_train_mean = predictions['gt_train'].double().mean()
    pred_val_mean = predictions['pred_val'].mean()
    gt_val_mean = predictions['gt_val'].double().mean()
    print(f"pred_train_mean {pred_train_mean:.3f},  gt_train_mean {gt_train_mean:.3f},  pred_val_mean {pred_val_mean:.3f},  gt_val_mean {gt_val_mean:.3f}")

    for i in range(-10, 10):
        minval_a = (i + 0) / 10
        minval_b = (i + 1) / 10

        #minval_a = gt_train_mean - (i + 1) / 20
        #minval_b = gt_train_mean - (i + 0) / 20

        print(f"minval {minval_a: 2.2f} - {minval_b: 2.2f}      ", end='')

        gt_train_mean = 0

        bigs_train = torch.logical_and(
            predictions['pred_train'] - gt_train_mean > minval_a,
            predictions['pred_train'] - gt_train_mean < minval_b
        ).nonzero().squeeze()

        bigs_val = torch.logical_and(
            predictions['pred_val'] - gt_train_mean > minval_a,
            predictions['pred_val'] - gt_train_mean < minval_b
        ).nonzero().squeeze()

        p_train = predictions['gt_train'][bigs_train]
        p_val = predictions['gt_val'][bigs_val]

        print_up_down(p_train)
        print("     ", end='')
        print_up_down(p_val)
        print()

    quit()

    quit()

    print_stats(pred)
    print_stats(gt.double())

    quit()


    print(pred[0:10])

    hy, hx = np.histogram(pred, bins=100)

    print(hy)
    print((hx[:-1] + hx[1:]) / 2)
    plt.plot((hx[:-1] + hx[1:]) / 2, hy)
    plt.grid(True)
    plt.show()
    quit()

    plt.hist(pred[0:10], bins=100)
    plt.show()

    quit()

    klines = pd.read_hdf(f"cache/klines/BTCUSDT.hdf")

    """
    class Portfolio:
        position_max_count = 5
        take_profit = 1.05
        stop_loss = 0.95

        def __init__(self):
            self.positions = []

        def add_position(self, symbol: str, position_size: float, mark_price: float):
            position = {
                'symbol': symbol,
                'size': position_size,
                'mark_price': mark_price,
                'take_profit': mark_price * self.take_profit,
                'stop_loss': mark_price * self.stop_loss
            }
            self.positions.append(position)
            return position

        def save(self):
            with open(f"position.pickle", 'wb') as f:
                pickle.dump(self.positions, f)

        def load(self):
            with open(f"position.pickle", 'rb') as f:
                self.positions = pickle.load(f)
    """

    start_timestamp = datetime.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_timestamp = datetime.strptime("2021-07-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    print((end_timestamp - start_timestamp).total_seconds() / 60)

    quit()

    a = np.zeros((3, ))
    a[0] = 1
    a[1] = 0
    a[2] = -1
    print(np.all(a))
    quit()

    with open(f"cache/filtered_symbols.pickle", 'rb') as f:
        symbols = pickle.load(f)

    print(symbols)
    quit()

    for file_path in glob.glob("cache/klines/*.hdf"):
        symbol = os.path.basename(file_path).replace('.hdf', '')
        if symbol == "BTCUSDT":
            data = pd.read_hdf(file_path)
            timestamp_first = datetime.fromtimestamp(data['open_time'].iloc[0] / 1000, tz=timezone.utc)
            timestamp_last = datetime.fromtimestamp(data['open_time'].iloc[-1] / 1000, tz=timezone.utc)

            diff = timestamp_last - timestamp_first

            for idx in range(1, data.shape[0]):
                if data['open_time'].iloc[idx] - data['open_time'].iloc[idx - 1] != 60000:
                    ts1 = datetime.fromtimestamp(data['open_time'].iloc[idx - 1] / 1000, tz=timezone.utc)
                    ts2 = datetime.fromtimestamp(data['open_time'].iloc[idx] / 1000, tz=timezone.utc)
                    print("Not 60 seconds:", ts1, ts2)

            print(diff.days * 24 * 60 + diff.seconds // 60)
            print()

    quit()

    pf = Portfolio()
    pf.load()

    #pf.add_position(symbol='s1', position_size=1.0, mark_price=1.1)
    #pf.add_position(symbol='s2', position_size=2.0, mark_price=2.1)
    #pf.add_position(symbol='s3', position_size=3.0, mark_price=3.1)

    #pf.save()

    quit()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.Formatter.converter = time.gmtime

    position = {
            'symbol': 1,
            'size': 2,
            'mark_price': 3,
            'take_profit': 4,
            'stop_loss': 5
        }
    logging.info(f"Hej {position}")
    quit()

    with open('binance_account.json') as f:
        account_info = json.load(f)

    symbols = ['BTCUSDT', 'ADAUSDT']

    binance_account = BinanceAccount(
        api_key=account_info['api_key'],
        api_secret=account_info['api_secret'],
        symbols=symbols
    )

    from time import sleep
    while True:
        sleep(1)

    quit()


    with open(f"cache/filtered_symbols.pickle", 'rb') as f:
        data = pickle.load(f)
    print(data)
    quit()

    btc_klines = pd.read_hdf(f"cache/klines/BTCUSDT.hdf")
    btc_klines['open_time'] = pd.to_datetime(btc_klines['open_time'], unit='ms')

    log = pd.read_csv(f"log_2021-06-11_05_00020c.txt", parse_dates=['time'], infer_datetime_format=True)

    # 2021-05-31 23:58:00+00:00

    plt.plot(log['time'], log['value']*10)
    plt.plot(btc_klines['open_time'], btc_klines['close'])
    plt.show()
    print(log)

    quit()

    with open(f"cache/training_data.pickle", 'rb') as f:
        data = pickle.load(f)

    i = data['input'][:, 0:1000]

    quit()

    with open(f"cache/intrinsic_events.pickle", 'rb') as f:
        intrinsic_events = pickle.load(f)

    print(intrinsic_events)
    quit()

    start_timestamp = datetime.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_timestamp = datetime.strptime("2021-05-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    n_timesteps = int((end_timestamp - start_timestamp).total_seconds() / 60)
    print(n_timesteps)

    symbol = 'BTCUSDT'
    klines = pd.read_hdf(f"cache/klines/{symbol}.hdf")
    print(klines.shape)

    with open(f"cache/training_data.pickle", 'rb') as f:
        data = pickle.load(f)

    print(data)
    print(data)


if __name__ == '__main__':
    main()
