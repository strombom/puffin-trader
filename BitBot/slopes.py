import pickle

import numba
import numpy as np
import pandas as pd
import multiprocessing
from scipy import stats


@numba.njit()  # nogil=True, parallel=True)
def lin_regress(x, y):
    a = np.vstack((np.ones_like(x), x)).T
    intercept, slope = np.linalg.lstsq(a, y)[0]
    return intercept, slope


@numba.njit()
def estimate_slope(slope_length: int, prices: np.ndarray, x_range: np.ndarray) -> tuple:
    # r = stats.linregress(x_range, prices[prices.shape[0] - slope_length:])
    # print(r)
    intercept, slope = lin_regress(x_range, prices[prices.shape[0] - slope_length:])
    slope_y = x_range * slope + intercept
    y_start, y_end = slope_y[0], slope_y[-1]
    s = prices.shape[0]
    p = prices[s - slope_length:]
    max_d = np.max(np.abs(p - slope_y))
    return y_start, y_end, max_d


#@numba.jit()
def make_slope(prices: np.ndarray, min_slope_length: int, max_slope_length: int, offset: int = 0) -> dict:
    volatilities = []
    slopes = []
    mean_price = np.mean(prices)
    x_range_all = np.arange(max_slope_length, dtype=np.float)
    for slope_length in range(max_slope_length, min_slope_length - 1, -1):
        x_range = x_range_all[:slope_length]
        y_start, y_end, volatility = estimate_slope(slope_length=slope_length, prices=prices, x_range=x_range)
        slopes.append((y_start, y_end, slope_length, volatility / 250))
        volatilities.append(volatility / mean_price)

    volatilities = np.array(volatilities) * 250
    x_range = np.arange(volatilities.shape[0], dtype=np.float)
    intercept, slope = lin_regress(x_range, volatilities)
    slope_y = x_range * slope + intercept
    volatilities = volatilities - slope_y

    y_start, y_end, slope_length, volatility = slopes[int(np.argmin(volatilities))]

    return {'length': slope_length / max_slope_length,
            'angle': 200 * (y_end - y_start) / prices[-1] / slope_length,
            'x0': offset - slope_length,
            'x1': offset - 1,
            'y0': y_start,
            'y1': y_end,
            'volatility': volatility}


class Slopes:
    min_slope_length = 10
    max_slope_length = 70

    # The latest price is not included in the slope
    cache_filename = f"cache/slopes.pickle"

    def __init__(self, prices: np.ndarray, use_cache=True):
        if use_cache:
            try:
                with open(self.cache_filename, 'rb') as f:
                    self.slopes = pickle.load(f)
                    if self.slopes.shape[0] == prices.shape[0] - self.max_slope_length:
                        return
            except FileNotFoundError:
                pass

        slopes = []
        for slope_idx in range(self.max_slope_length, prices.shape[0]):
            slope = make_slope(prices[slope_idx - self.max_slope_length:slope_idx], self.min_slope_length, self.max_slope_length, slope_idx)
            slope['idx'] = slope_idx
            slopes.append(slope)
        self.slopes = pd.DataFrame(slopes)

        with open(self.cache_filename, 'wb') as f:
            pickle.dump(self.slopes, f, pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, item):
        return self.slopes[item]

    def __len__(self):
        return len(self.slopes)
