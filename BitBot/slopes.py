import pickle

import numpy as np
from scipy import stats


class Slopes:
    # The latest price is not included in the slope
    max_slope_length = 70
    min_slope_length = 10
    x_range = np.arange(max_slope_length)
    cache_filename = f"cache/slopes.pickle"

    def __init__(self, prices: np.ndarray, use_cache=True):
        if use_cache:
            try:
                with open(self.cache_filename, 'rb') as f:
                    data = pickle.load(f)
                    if data['slope_count'] == prices.shape[0] - self.max_slope_length:
                        self.slopes = data['slopes']
                        return
            except FileNotFoundError:
                pass

        self.slopes = []
        for slope_idx in range(self.max_slope_length, prices.shape[0]):
            self.slopes.append(Slope(prices=prices[slope_idx - self.max_slope_length:slope_idx], offset=slope_idx))

        with open(self.cache_filename, 'wb') as f:
            data = {'slope_count': len(self.slopes),
                    'slopes': self.slopes}
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, item):
        return self.slopes[item]

    def __len__(self):
        return len(self.slopes)


class Slope:
    def __init__(self, prices: np.ndarray, offset: int):
        (y_start, y_end, slope_length, volatility), volatilities = self.find_best_fit(prices)
        self.length = slope_length / Slopes.max_slope_length
        self.angle = 200 * (y_end - y_start) / prices[-1] / slope_length
        self.x = np.array((offset - slope_length, offset - 1))
        self.y = np.array((y_start, y_end))
        self.volatility = volatility
        self.volatilities = volatilities

    def find_best_fit(self, prices: np.ndarray) -> tuple:
        volatilities = []
        slopes = []
        mean_price = np.mean(prices)
        for slope_length in range(Slopes.max_slope_length, Slopes.min_slope_length - 1, -1):
            y_start, y_end, volatility = self.estimate_slope(slope_length=slope_length, prices=prices)
            slopes.append((y_start, y_end, slope_length, volatility / 250))
            volatilities.append(volatility / mean_price)
        volatilities = np.array(volatilities) * 250
        x_range = np.arange(volatilities.shape[0])
        r = stats.linregress(x_range, volatilities)
        slope_y = x_range * r.slope + r.intercept
        volatilities = volatilities - slope_y
        best_slope = slopes[int(np.argmin(volatilities))]
        return best_slope, volatilities

    def estimate_slope(self, slope_length: int, prices: np.ndarray) -> tuple:
        x_range = Slopes.x_range[:slope_length]
        r = stats.linregress(x_range, prices[prices.shape[0] - slope_length:])
        slope_y = x_range * r.slope + r.intercept
        y_start, y_end = slope_y[0], slope_y[-1]
        s = prices.shape[0]
        p = prices[s - slope_length:]
        max_d = np.max(np.abs(p - slope_y))
        return y_start, y_end, max_d
