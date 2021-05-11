
import scipy
import pickle
import numpy as np
import pandas as pd
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
    file_path = f"cache/klines/{symbol}.hdf"
    klines = pd.DataFrame(pd.read_hdf(file_path))

    data_length = 0
    end_epoch = end_timestamp.timestamp() * 1000
    for idx in reversed(klines.index):
        if klines.open_time[idx] <= end_epoch:
            data_length = idx
            break

    return data_length


def make_steps(symbols: list, end_timestamp: datetime):
    #steps_file_path = f"cache/steps.npy"
    #try:
    #    steps = np.load(file=steps_file_path)
    #    return steps
    #except FileNotFoundError:
    #    pass

    data_length = get_data_length(symbol=symbols[0], end_timestamp=end_timestamp)

    data_length = min(12000, data_length)
    symbols = symbols[:1]

    deltas = np.array([0.002, 0.004, 0.008, 0.016, 0.032, 0.064])
    steps = np.zeros((len(symbols), deltas.shape[0], data_length))
    steps_timestamps = None

    for symbol_idx, symbol in enumerate(symbols):
        file_path = f"cache/klines/{symbol}.hdf"
        klines = pd.DataFrame(pd.read_hdf(file_path))
        klines = klines[:data_length]

        if steps_timestamps is None:
            steps_timestamps = klines['open_time'][:data_length].to_numpy() / 1000

        for delta_idx, delta in enumerate(deltas):
            runner = Runner(delta=delta)
            for kline_idx, row in klines.iterrows():
                ie_events = runner.step(row['close'])
                steps[symbol_idx, delta_idx, kline_idx] = len(ie_events)

    #np.save(file=steps_file_path, arr=steps)
    return steps_timestamps, steps


def calc_weekly_steps(steps_timestamps: np.ndarray, symbols: list, steps: np.ndarray):
    start_timestamp = steps_timestamps[0]

    for symbol in symbols:
        current_week = datetime.utcfromtimestamp(start_timestamp)
        current_week -= timedelta(days=current_week.weekday() % 7)
        next_week = current_week + timedelta(weeks=1)

        for kline_idx in range(steps_timestamps.shape[0]):
            if steps_timestamps[kline_idx] > next_week:
                print("a")
            print(symbol)

    return steps


def main():
    with open(f"cache/filtered_symbols.pickle", 'rb') as f:
        symbols = pickle.load(f)

    end_timestamp = datetime.strptime("2021-05-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    steps_timestamps, steps = make_steps(symbols=symbols, end_timestamp=end_timestamp)
    weekly_steps = calc_weekly_steps(steps_timestamps=steps_timestamps, symbols=symbols, steps=steps)

    print(steps)
    quit()

    for symbol in symbols:

        smooth_volatility = calc_smooth_volatility(data)
        """
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(smooth_volatility)
        plt.title(symbol)
        plt.show()
        """

        def curve_fit(x, n_steps_target):
            n_steps = 1

            runner_trade = Runner(delta=0)
            for idx, price in enumerate(data['close']):
                runner_trade.delta = x[0] * smooth_volatility[idx] + x[1]
                trade_prices = runner_trade.step(price=price)
                n_steps += len(trade_prices)

            return n_steps - n_steps_target

        ass = []
        bss = []
        stepss = []
        for a in np.arange(start=0.02, stop=0.2, step=0.01):
            for b in np.arange(start=0.00, stop=0.006, step=0.001):
                n_steps = curve_fit(x=(a, b), n_steps_target=0)
                print(round(a, 3), round(b, 3), n_steps)
                ass.append(a)
                bss.append(b)
                stepss.append(n_steps)

        with open(f"cache/analysis.pickle", 'wb') as f:
            pickle.dump((ass, bss, stepss), f, pickle.HIGHEST_PROTOCOL)

        quit()
        res = scipy.optimize.least_squares(fun=curve_fit, x0=[0.1, 0.0], args=(5000, ))

        print(sum(smooth_volatility))

        deltas = np.array([0.002, 0.004, 0.008, 0.016, 0.032, 0.064])

        import matplotlib.pyplot as plt
        plt.clf()
        plt.ylim((0, 1))
        plt.plot(smooth_volatility)
        # plt.plot(b)
        plt.title(symbol)
        print(f"Save {symbol}")
        plt.savefig(fname=f"tmp/steps_{symbol}.png")

        """
        def f(x):
            runner_trade = Runner(delta=0.005)
            count = 0

            for step_idx, step_value in enumerate(step_values):
                runner_trade.delta = x[0] * step_value + 0.001
                trade_prices = runner_trade.step(price=price)
                count += len(trade_prices)

            print(x, count)
            return 40000 - count

        res = least_squares(fun=f, x0=[0.005])

        print(res)
        print()
        """

        """
        deltas = np.array([0.005, 0.007, 0.01, 0.02, 0.03, 0.05])
        steps = np.zeros((data.shape[0], deltas.shape[0]))

        for delta_idx, delta in enumerate(deltas):
            runner = Runner(delta=delta)

            for time_idx in range(data.shape[0]):
                ie_prices = runner.step(price=data['close'][time_idx])
                steps[time_idx, delta_idx] = len(ie_prices)

        steps = pd.DataFrame(data=steps, columns=deltas)
        s = steps[deltas[0]]
        a = s.ewm(alpha=1/1000).mean()

        import matplotlib.pyplot as plt

        plt.plot(a)
        # plt.plot(b)
        plt.show()

        print(symbol, steps.sum(axis=0))
        # print()
        """

        """
        ema = ExponentialMovingAverage(ema_weight_up=1/1000, ema_weight_down=1/1000, initial_value=0.0)
        
        b = np.empty((steps[deltas[0]].shape[0]))
        for idx, price in enumerate(s):
            b[idx] = ema.step(price)
        # b = s.ewm(halflife=500).mean()
        a /= np.max(a)
        b /= np.max(b)
        # b -= 0.1

        """

        # break


if __name__ == '__main__':
    main()
