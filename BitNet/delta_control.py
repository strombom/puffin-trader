import pickle
import numpy as np
import pandas as pd
from simple_pid import PID
import matplotlib.pyplot as plt

from cache import cache_it
from IntrinsicTime.runner import Runner


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
    symbol = "DASHUSDT"

    intrinsic_events = {}

    for symbol in symbols:
        print(symbol)
        #timesteps = make_timesteps(symbol=symbol, delta=delta)
        pid_result = make_pid_timesteps(symbol=symbol, delta=delta, setpoint=30)
        intrinsic_events[symbol] = pid_result

        #break

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
    axs[0].plot(pid_result['prices'])
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
