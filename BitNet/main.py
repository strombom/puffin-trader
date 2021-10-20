import queue
import pickle
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt

from IntrinsicTime.runner import Runner
from cache import cache_it


@cache_it
def make_steps(symbol: str, delta: float) -> (list, list):
    runner = Runner(delta=delta)
    timestamps, steps = [], []
    klines = pd.read_hdf(f"cache/klines/{symbol}.hdf")
    for kline_idx, kline in klines.iterrows():
        mark_price = kline['close']
        for step in runner.step(mark_price):
            timestamps.append(kline_idx)
            steps.append(step)
    return timestamps, steps


@cache_it
def mark_klines(symbol: str, take_profit: float, stop_loss: float):
    print("mark_klines", symbol, take_profit, stop_loss)
    klines = pd.read_hdf(f"cache/klines/{symbol}.hdf")
    positions = {}
    profits = np.empty(klines.shape[0])
    for kline_idx, kline in klines.iterrows():
        mark_price = kline['close']
        for position_idx in list(positions.keys()):
            position = positions[position_idx]
            if mark_price >= position['take_profit']:
                profits[position_idx] = 1
                del position
            elif mark_price <= position['stop_loss']:
                profits[position_idx] = -1
                del position

        positions[kline_idx] = {
            'take_profit': mark_price * take_profit,
            'stop_loss': mark_price * stop_loss
        }

    print(profits)


@cache_it
def mark_steps(steps: pd.DataFrame, take_profit: float, stop_loss: float):
    print("mark_steps", take_profit, stop_loss)
    positions = {}
    profits = np.zeros(steps.shape[0])
    for step_idx, step in steps.iterrows():
        mark_price = step['price']
        for position_idx in list(positions.keys()):
            if mark_price >= positions[position_idx]['take_profit']:
                profits[position_idx] = 1
                del positions[position_idx]
            elif mark_price <= positions[position_idx]['stop_loss']:
                profits[position_idx] = -1
                del positions[position_idx]

        positions[step_idx] = {
            'take_profit': mark_price * take_profit,
            'stop_loss': mark_price * stop_loss
        }

    return profits


def mark_klines_process(task_queue):
    while True:
        try:
            symbol, take_profit, stop_loss = task_queue.get(block=False)
        except queue.Empty:
            break
        mark_klines(symbol, take_profit, stop_loss)


def main():
    with open('cache/filtered_symbols.pickle', 'rb') as f:
        symbols = pickle.load(f)

    deltas = (0.20, 0.10, 0.05, 0.02, 0.01, 0.005)
    symbol = 'BTCUSDT'

    steps = {symbol: {}}
    for delta in deltas:
        sym_timestamps, sym_steps = make_steps(symbol=symbol, delta=delta)
        steps[symbol][delta] = pd.DataFrame({
            'timestamp': np.array(sym_timestamps),
            'price': np.array(sym_steps)
        })

    limits = [
        (1.05, 0.80), (1.05, 0.85), (1.05, 0.90), (1.05, 0.95),
        (1.10, 0.80), (1.10, 0.85), (1.10, 0.90), (1.10, 0.95),
        (1.15, 0.80), (1.15, 0.85), (1.15, 0.90), (1.15, 0.95),
        (1.20, 0.80), (1.20, 0.85), (1.20, 0.90), (1.20, 0.95)
    ]

    fee = 0.075 / 100 * 2
    delta = 0.01

    for take_profit, stop_loss in limits:
        profits = mark_steps(steps=steps[symbol][delta], take_profit=take_profit, stop_loss=stop_loss)

    for take_profit, stop_loss in limits:
        print(f"tp:{take_profit} sl:{stop_loss}")
        profits = mark_steps(steps=steps[symbol][delta], take_profit=take_profit, stop_loss=stop_loss)
        fig, axs = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        axs[0].plot(steps[symbol][delta]['price'])
        axs[0].set_yscale('log')
        axs[1].plot(profits)
        plt.show()


    """
    task_queue = multiprocessing.Queue()

    for take_profit, stop_loss in limits:
        task_queue.put((symbol, take_profit, stop_loss))

    ps = []
    for n in range(4):
        # for n in range(min(multiprocessing.cpu_count() - 3, len(symbols))):
        # mark_klines(symbol=symbol, take_profit=take_profit, stop_loss=stop_loss)
        p = multiprocessing.Process(target=mark_klines, args=(task_queue, ))
        ps.append(p)
        p.start()

    for p in ps:
        p.join()
    """

    print(limits)


    for delta in deltas:
        timestamps, steps = make_steps(symbol=symbol, delta=delta)
        plt.plot(timestamps, steps)
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    main()
