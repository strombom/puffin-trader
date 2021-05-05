import os
import glob
import pickle
import numpy as np
import pandas as pd
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


def main():
    with open(f"cache/filtered_symbols.pickle", 'rb') as f:
        symbols = pickle.load(f)

    for symbol in symbols:
        file_path = f"cache/klines/{symbol}.hdf"
        data = pd.DataFrame(pd.read_hdf(file_path))

        # data = data[:200000]
        data = data[:100000]

        smooth_volatility = calc_smooth_volatility(data)


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
