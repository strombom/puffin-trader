import pickle
import queue
import time

import numpy as np
import pandas as pd
import multiprocessing
from simple_pid import PID
import matplotlib.pyplot as plt

from cache import cache_it
from IntrinsicTime.runner import Runner


#@cache_it
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
        timestamps, steps = [], []
        klines = pd.read_hdf(f"cache/klines/{symbol}.hdf")
        for kline_idx, kline in klines.iterrows():
            mark_price = kline['close']
            for step in runner.step(mark_price):
                timestamps.append(kline_idx)
                steps.append(step)
            #if len(steps) > 1000:
            #    break

        with results_queue_lock:
            results_queue.put({
                'symbol': parameters['symbol'],
                'timestamps': timestamps,
                'steps': steps
            })


@cache_it
def make_timesteps(symbol: str, delta: float):
    runner = Runner(delta=delta)
    timesteps = []
    klines = pd.read_hdf(f"cache/klines/{symbol}.hdf")
    kline_idx_prev = 0
    for kline_idx, kline in klines.iterrows():
        mark_price = kline['close']
        for step in runner.step(mark_price):
            timestep = kline_idx - kline_idx_prev
            kline_idx_prev = kline_idx
            timesteps.append(timestep)
    return timesteps


@cache_it
def make_pid_timesteps(symbol: str, delta: float, setpoint: float):
    pid = PID(Kp=1.0, Ki=2.0, Kd=0.00, setpoint=setpoint)
    runner = Runner(delta=delta)
    timesteps, deltas = [], []
    timestamps, prices = [], []
    klines = pd.read_hdf(f"cache/klines/{symbol}.hdf")
    kline_idx_prev = 0
    for kline_idx, kline in klines.iterrows():
        mark_price = kline['close']
        for step in runner.step(mark_price):
            timestep = kline_idx - kline_idx_prev
            control = pid(input_=timestep, dt=1)
            new_delta = delta * (1 + control / 1000)
            new_delta = max(new_delta, 0.0025)
            new_delta = min(new_delta, 0.025)
            runner.delta = new_delta
            kline_idx_prev = kline_idx
            timesteps.append(timestep)
            deltas.append(new_delta)
            prices.append(step)
            timestamps.append(kline_idx)
    return {
        'timesteps': timesteps,
        'deltas': deltas,
        'timestamps': timestamps,
        'prices': prices
    }


def main():
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
        print("New thread", n)
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
                    'timestamps': result['timestamps'],
                    'steps': result['steps']
                }
        except queue.Empty:
            time.sleep(0.1)

    #for p in ps:
    #    p.join()
    #    print("joined")

    """
    intrinsic_events = {}
    while True:
        try:
            steps = results_queue.get(block=True, timeout=1)
            print("got", steps['symbol'])
            intrinsic_events[steps['symbol']] = {
                'timestamps': steps['timestamps'],
                'steps': steps['steps']
            }

        except queue.Empty:
            print("all done")
            break
    """

    #print(intrinsic_events)
    """
    for symbol in symbols:
        print(symbol)
        intrinsic_events[symbol] = make_steps(symbol=symbol, delta=delta)

        #pid_result = make_pid_timesteps(symbol=symbol, delta=delta, setpoint=30)
        #intrinsic_events[symbol] = pid_result

    """

    with open(f"cache/intrinsic_events.pickle", 'wb') as f:
        pickle.dump(intrinsic_events, f, pickle.HIGHEST_PROTOCOL)

    # print("median", np.median(timesteps))
    # print("average", np.average(timesteps))

    #plt.hist(timesteps, bins=500)
    #plt.hist(pid_timesteps, bins=500)

    #plt.plot(timesteps)
    #plt.plot(pid_timesteps)

    fig, axs = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    #axs[0].plot(timesteps)
    axs[0].plot(intrinsic_events['BTCUSDT']['steps'])
    axs[0].set_yscale('log')
    #axs[1].plot(pid_result['deltas'])
    plt.show()


    """
    steps = make_steps(symbol=symbol, delta=delta)
    fig, axs = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    axs[0].plot(steps[1])
    axs[0].set_yscale('log')
    plt.show()
    """


if __name__ == '__main__':
    main()
