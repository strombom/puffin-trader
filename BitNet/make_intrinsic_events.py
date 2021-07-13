import queue
import pickle
import time
from datetime import datetime, timezone

import pandas as pd
import multiprocessing

from matplotlib import pyplot as plt

from IntrinsicTime.runner import Runner


def make_ie_process(task_queue_lock, task_queue, results_queue_lock, results_queue, process_id):
    while True:
        try:
            with task_queue_lock:
                parameters = task_queue.get(block=True, timeout=1)
        except queue.Empty:
            break

        symbol = parameters['symbol']
        delta = parameters['delta']
        print(process_id, "Processing", symbol)

        runner = Runner(delta=delta)
        timestamps, prices = [], []
        klines = pd.read_hdf(f"cache/klines/{symbol}.hdf")
        for kline_idx, kline in klines.iterrows():
            mark_price = kline['open']
            for ie_price in runner.step(mark_price):
                timestamps.append(datetime.fromtimestamp(kline['open_time'] / 1000, tz=timezone.utc))
                prices.append(ie_price)

        with results_queue_lock:
            results_queue.put({
                'symbol': parameters['symbol'],
                'timestamps': timestamps,
                'prices': prices
            })


def make_intrinsic_events(delta: float):
    with open(f"cache/filtered_symbols.pickle", 'rb') as f:
        symbols = pickle.load(f)

    print(f"make_intrinsic_events: {symbols}")

    task_queue = multiprocessing.Queue()
    task_queue_lock = multiprocessing.Lock()

    results_queue = multiprocessing.Queue()
    results_queue_lock = multiprocessing.Lock()

    for symbol in symbols:
        task_queue.put({
            'symbol': symbol,
            'delta': delta
        })

    ps = []
    for n in range(min(multiprocessing.cpu_count() - 3, len(symbols))):
        p = multiprocessing.Process(
            target=make_ie_process,
            args=(task_queue_lock, task_queue, results_queue_lock, results_queue, n)
        )
        ps.append(p)
        p.start()

    intrinsic_events = {}
    while True:
        if not any(p.is_alive() for p in ps):
            break

        try:
            with task_queue_lock:
                result = results_queue.get(block=True, timeout=1)
                intrinsic_events[result['symbol']] = {
                    'timestamps': result['timestamps'],
                    'prices': result['prices']
                }
        except queue.Empty:
            time.sleep(0.1)

    with open(f"cache/intrinsic_events.pickle", 'wb') as f:
        pickle.dump(intrinsic_events, f, pickle.HIGHEST_PROTOCOL)

    fig, axs = plt.subplots(nrows=2, sharex='row', gridspec_kw={'height_ratios': [4, 1]})
    #axs[0].plot(timesteps)
    axs[0].plot(intrinsic_events['BTCUSDT']['prices'])
    axs[0].set_yscale('log')
    #axs[1].plot(pid_result['deltas'])
    plt.show()


if __name__ == '__main__':
    make_intrinsic_events(delta=0.0025)
