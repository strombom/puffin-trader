import queue
import pickle
import time

import pandas as pd
import multiprocessing

from matplotlib import pyplot as plt

from IntrinsicTime.runner import Runner


def make_steps_process(task_queue_lock, task_queue, results_queue_lock, results_queue, process_id):
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
        kline_idxs, step_prices = [], []
        klines = pd.read_hdf(f"cache/klines/{symbol}.hdf")
        for kline_idx, kline in klines.iterrows():
            mark_price = kline['open']
            for step_price in runner.step(mark_price):
                kline_idxs.append(kline_idx)
                step_prices.append(step_price)

        with results_queue_lock:
            results_queue.put({
                'symbol': parameters['symbol'],
                'kline_idxs': kline_idxs,
                'step_prices': step_prices
            })


def make_runner_events():
    with open(f"cache/filtered_symbols.pickle", 'rb') as f:
        symbols = pickle.load(f)

    print(symbols)
    delta = 0.01

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
        p = multiprocessing.Process(target=make_steps_process, args=(
            task_queue_lock, task_queue, results_queue_lock, results_queue, n)
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
                    'kline_idxs': result['kline_idxs'],
                    'step_prices': result['step_prices']
                }
        except queue.Empty:
            time.sleep(0.1)

    with open(f"cache/intrinsic_events.pickle", 'wb') as f:
        pickle.dump(intrinsic_events, f, pickle.HIGHEST_PROTOCOL)

    fig, axs = plt.subplots(nrows=2, sharex='row', gridspec_kw={'height_ratios': [4, 1]})
    #axs[0].plot(timesteps)
    axs[0].plot(intrinsic_events['BTCUSDT']['step_prices'])
    axs[0].set_yscale('log')
    #axs[1].plot(pid_result['deltas'])
    plt.show()


if __name__ == '__main__':
    make_runner_events()
