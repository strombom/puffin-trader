import queue

import scipy
import pickle
import scipy.signal
import numpy as np
import pandas as pd
import multiprocessing
from datetime import datetime, timezone, timedelta
from scipy.optimize import least_squares

from IntrinsicTime.runner import Runner


class ExponentialMovingAverage:
    def __init__(self, ema_weight_up: float, ema_weight_down: float, initial_value: float):
        self.ema_weight_up = ema_weight_up
        self.ema_weight_down = ema_weight_down
        self.ema_value = initial_value

    def step(self, value):
        if value > self.ema_value:
            self.ema_value = self.ema_weight_up * value + (1 - self.ema_weight_up) * self.ema_value
        else:
            self.ema_value = self.ema_weight_down * value + (1 - self.ema_weight_down) * self.ema_value

        return self.ema_value


def calc_smooth_volatility(data):
    smooth_volatility = np.zeros([data.shape[0]])
    clock_ema = ExponentialMovingAverage(ema_weight_up=1 / 100000, ema_weight_down=1 / 100000, initial_value=0.10)
    runner_clock = Runner(delta=0.005)
    for time_idx, price in enumerate(data['close']):
        clock_prices = runner_clock.step(price=price)
        ema_value = clock_ema.step(value=len(clock_prices))
        smooth_volatility[time_idx] = ema_value
    return smooth_volatility


def get_data_length(symbol: str, end_timestamp: datetime):
    data_length_file_path = f"cache/tmp_get_data_length.pickle"
    try:
        with open(data_length_file_path, 'rb') as f:
            data_length = pickle.load(f)
            return data_length
    except FileNotFoundError:
        pass

    file_path = f"cache/klines/{symbol}.hdf"
    klines = pd.DataFrame(pd.read_hdf(file_path))

    data_length = 0
    end_epoch = end_timestamp.timestamp() * 1000
    for idx in reversed(klines.index):
        if klines.open_time[idx] <= end_epoch:
            data_length = idx
            break

    with open(data_length_file_path, 'wb') as f:
        pickle.dump(data_length, f)

    return data_length


def process_steps(task_queue_, result_queue_, data_length, deltas, process_id):
    steps_timestamps = None
    print(f"p start {process_id}")
    while True:
        try:
            symbol_idx, symbol = task_queue_.get(block=False)
        except queue.Empty:
            break

        print(f"Process[{process_id}] {symbol} start")

        file_path = f"cache/klines/{symbol}.hdf"
        klines = pd.DataFrame(pd.read_hdf(file_path))
        klines = klines[:data_length]

        if steps_timestamps is None:
            steps_timestamps = klines['open_time'][:data_length].to_numpy() / 1000

        steps = np.zeros((data_length, deltas.shape[0]))
        for delta_idx, delta in enumerate(deltas):
            runner = Runner(delta=delta)
            for kline_idx, row in klines.iterrows():
                ie_events = runner.step(row['close'])
                steps[kline_idx, delta_idx] = len(ie_events)

        result_queue_.put((symbol_idx, steps))
        print(f"Process[{process_id}] {symbol} done")

    print(f"p stop {process_id}")


def make_steps(symbols: list, end_timestamp: datetime, deltas: np.ndarray):
    steps_file_path = f"cache/steps.pickle"
    try:
        with open(steps_file_path, 'rb') as f:
            steps_timestamps, steps = pickle.load(f)
            return steps_timestamps, steps
    except FileNotFoundError:
        pass

    data_length = get_data_length(symbol=symbols[0], end_timestamp=end_timestamp)

    # data_length = min(100000, data_length)
    # symbols = symbols[:1]

    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    #task_queue.put((0, symbols[0]))
    #process_steps(task_queue, result_queue, data_length, deltas)

    for symbol_idx, symbol in enumerate(symbols):
        task_queue.put((symbol_idx, symbol))

    processes = []
    for n in range(min(multiprocessing.cpu_count() - 3, len(symbols))):
        p = multiprocessing.Process(target=process_steps, args=(task_queue, result_queue, data_length, deltas, n))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    steps = np.zeros((len(symbols), data_length, deltas.shape[0]))
    while not result_queue.empty():
        symbol_idx, symbol_steps = result_queue.get()
        print(f"Got result {symbol_idx}")
        steps[symbol_idx] = symbol_steps

    with open(steps_file_path, 'wb') as f:
        pickle.dump((steps_timestamps, steps), f)

    return steps_timestamps, steps


def calc_accum_steps(steps_timestamps: np.ndarray, symbols: list, steps: np.ndarray, deltas: np.ndarray):
    lookback_length = timedelta(weeks=1)
    period_length = timedelta(days=1)

    start_timestamp = steps_timestamps[0]
    accum_steps = {}

    for symbol_idx in range(steps.shape[0]):
        symbol = symbols[symbol_idx]
        print("Calc accum steps", symbol)
        current_date = datetime.fromtimestamp(start_timestamp, tz=timezone.utc) + lookback_length

        kline_idx_start = 0

        def find_kline(kline_idx, target_date):
            target_timestamp = target_date.timestamp()
            while kline_idx < steps_timestamps.shape[0] and steps_timestamps[kline_idx] < target_timestamp:
                kline_idx += 1
            return kline_idx

        kline_idx_end = find_kline(kline_idx_start, current_date)

        while kline_idx_end < steps_timestamps.shape[0]:
            if current_date not in accum_steps:
                accum_steps[current_date] = {}

            accum_steps[current_date][symbol] = steps[symbol_idx, kline_idx_start:kline_idx_end].sum(axis=0)
            current_date += period_length
            kline_idx_start = kline_idx_end
            kline_idx_end = find_kline(kline_idx_start, current_date)

    return accum_steps


def optimize_delta(accum_steps: dict, deltas: np.ndarray):
    target = 1000
    optimized_deltas = {}

    def curve_fun(x, a, b, c):
        return a / np.log1p(b * x) + c

    def curve_fit(x, y):
        p_opt, _ = scipy.optimize.curve_fit(curve_fun, x, y, p0=[-5.24722065e-01, -9.76472032e-03, -6.61553567e+02])
        return p_opt

    for date in accum_steps:
        print("Optimize delta", date)
        optimized_deltas[date] = {}
        for symbol in accum_steps[date]:
            popt = curve_fit(deltas, accum_steps[date][symbol])
            res = scipy.optimize.minimize_scalar(
                lambda x: abs(curve_fun(x, *popt) - target),
                bounds=(0.0002, 0.2),
                method='bounded'
            )
            optimized_deltas[date][symbol] = res.x

    return optimized_deltas


def moving_median(data_set, periods=5):
    import scipy.signal
    symbols = list(data_set[next(iter(data_set))].keys())
    values = np.empty((len(symbols), len(data_set)))

    for symbol_idx_, symbol_ in enumerate(symbols):
        for date_idx, date_ in enumerate(data_set):
            values[symbol_idx_][date_idx] = data_set[date_][symbol_]

    for symbol_idx_, symbol_ in enumerate(symbols):
        offset = periods // 2
        filtered = scipy.signal.medfilt(values[symbol_idx_], kernel_size=periods)
        filtered = np.concatenate((
            np.ones(offset) * filtered[0],
            filtered[0:filtered.shape[0] - offset]
        ))
        values[symbol_idx_] = filtered

    for symbol_idx_, symbol_ in enumerate(symbols):
        for date_idx, date_ in enumerate(data_set):
            data_set[date_][symbol_] = values[symbol_idx_][date_idx]

    return data_set


def main():
    with open(f"cache/filtered_symbols.pickle", 'rb') as f:
        symbols = pickle.load(f)

    # symbols = ['ADAUSDT', 'BTCUSDT', 'FTMUSDT', 'ETHUSDT']

    deltas = np.array([0.001, 0.003, 0.006, 0.016, 0.032, 0.064])

    end_timestamp = datetime.strptime("2021-05-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    steps_timestamps, steps = make_steps(symbols=symbols, end_timestamp=end_timestamp, deltas=deltas)

    avg = np.average(steps, axis=1)
    accum_steps = calc_accum_steps(steps_timestamps=steps_timestamps, symbols=symbols, steps=steps, deltas=deltas)
    optim_deltas = optimize_delta(accum_steps=accum_steps, deltas=deltas)

    optim_deltas = moving_median(optim_deltas)

    optim_deltas_file = f"cache/optim_deltas.pickle"
    with open(optim_deltas_file, 'wb') as f:
        pickle.dump(optim_deltas, f)

    print(optim_deltas)


if __name__ == '__main__':
    main()
