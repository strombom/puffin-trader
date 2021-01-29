import pickle

import numpy as np
from scipy import stats


class Slopes:
    # The latest price is not included in the slope
    max_slope_length = 70
    min_slope_length = 10
    x_range = np.arange(max_slope_length)

    def __init__(self, prices: np.ndarray):
        try:
            with open(f"cache/slopes.pickle", 'rb') as f:
                data = pickle.load(f)
                if data['slope_count'] == prices.shape[0] - self.max_slope_length:
                    self.slopes = data['slopes']
                    return
        except FileNotFoundError:
            pass

        self.slopes = []
        for slope_idx in range(self.max_slope_length, prices.shape[0]):
            self.slopes.append(Slope(prices=prices[slope_idx - self.max_slope_length:slope_idx], offset=slope_idx))

        with open(f"cache/slopes.pickle", 'wb') as f:
            data = {'slope_count': len(self.slopes),
                    'slopes': self.slopes}
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


class Slope:
    def __init__(self, prices: np.ndarray, offset: int):
        self.prices = prices
        y_start, y_end, slope_length = self.find_best_fit()
        self.length = slope_length
        self.angle = 1000 * (y_end - y_start) / prices[-1] / slope_length
        self.x = np.array((offset - slope_length, offset - 1))
        self.y = np.array((y_start, y_end))

    def find_best_fit(self):
        min_d = 1e9
        best_slope = None
        for slope_length in range(Slopes.min_slope_length, Slopes.max_slope_length + 1):
            y_start, y_end, max_d = self.estimate_slope(slope_length=slope_length)
            max_d = max_d / y_start * 100 - slope_length / 75
            if max_d < min_d:
                min_d = max_d
                best_slope = (y_start, y_end, slope_length)
        return best_slope

    def estimate_slope(self, slope_length):
        x_range = Slopes.x_range[:slope_length]
        r = stats.linregress(x_range, self.prices[self.prices.shape[0] - slope_length:])
        slope_y = x_range * r.slope + r.intercept
        y_start, y_end = slope_y[0], slope_y[-1]
        s = self.prices.shape[0]
        p = self.prices[s - slope_length:]
        max_d = np.argmax(np.abs(p - slope_y))
        return y_start, y_end, max_d
