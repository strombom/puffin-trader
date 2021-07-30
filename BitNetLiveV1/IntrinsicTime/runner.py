from enum import IntEnum

from numba import jit
from numba.experimental import jitclass
from numba import types
from numba.typed import List as NumbaList
from typing import List


#class Direction(IntEnum):
#    up = 1
#    down = -1

spec = [
    ('delta', types.float64),
    ('current_price', types.float64),
    ('previous_price', types.float64),
    ('ie_start_price', types.float64),
    ('ie_max_price', types.float64),
    ('ie_min_price', types.float64),
    ('ie_delta_top', types.float64),
    ('ie_delta_bot', types.float64),
    ('price', types.float64),
    ('initialised', types.boolean)
]


@jitclass(spec)
class Runner:
    def __init__(self, delta: float):
        self.delta = delta
        self.current_price = 0
        self.previous_price = 0
        self.ie_start_price = 0
        self.ie_max_price = 0
        self.ie_min_price = 0
        self.ie_delta_top = 0
        self.ie_delta_bot = 0
        self.initialised = False

    def step(self, price: float) -> List[float]:
        if not self.initialised:
            self.current_price = price
            self.previous_price = self.current_price
            self.ie_start_price = self.current_price
            self.ie_max_price = price
            self.ie_min_price = price
            self.initialised = True

        ie_prices = []

        if price > self.current_price:
            self.current_price = price
            # self.current_price = min(self.current_price, high)
        elif price < self.current_price:
            self.current_price = price
            # self.current_price = max(self.current_price, low)
        else:
            return ie_prices

        if self.current_price > self.previous_price:
            delta_dir = 1
        else:
            delta_dir = -1
        self.previous_price = self.current_price

        if self.current_price > self.ie_max_price:
            self.ie_max_price = self.current_price
            self.ie_delta_top = (self.ie_max_price - self.ie_start_price) / self.ie_start_price
        elif self.current_price < self.ie_min_price:
            self.ie_min_price = self.current_price
            self.ie_delta_bot = (self.ie_start_price - self.ie_min_price) / self.ie_start_price

        delta_down = (self.ie_max_price - self.current_price) / self.ie_max_price  # Delta from top
        delta_up = (self.current_price - self.ie_min_price) / self.ie_min_price    # Delta from bot

        if self.ie_delta_top + delta_down >= self.delta or self.ie_delta_bot + delta_up >= self.delta:
            if delta_dir == 1:
                remaining_delta = self.ie_delta_bot + delta_up
                ie_price = self.ie_min_price * (1.0 + (self.delta - self.ie_delta_bot))
            else:
                remaining_delta = self.ie_delta_top + delta_down
                ie_price = self.ie_max_price * (1.0 - (self.delta - self.ie_delta_top))

            while remaining_delta >= 2 * self.delta:
                if delta_dir == 1:
                    self.ie_max_price = min(self.ie_max_price, ie_price)
                else:
                    self.ie_min_price = max(self.ie_min_price, ie_price)

                ie_prices.append(ie_price)

                next_price = ie_price * (1.0 + delta_dir * self.delta)
                self.ie_start_price = ie_price
                if delta_dir == 1:
                    self.ie_max_price = next_price
                    self.ie_min_price = ie_price
                else:
                    self.ie_max_price = ie_price
                    self.ie_min_price = next_price
                self.ie_delta_top = (self.ie_max_price - self.ie_start_price) / self.ie_start_price
                self.ie_delta_bot = (self.ie_start_price - self.ie_min_price) / self.ie_start_price
                ie_price = next_price
                remaining_delta -= self.delta

            ie_prices.append(ie_price)

            self.ie_start_price = ie_price
            self.ie_max_price = ie_price
            self.ie_min_price = ie_price
            self.ie_delta_top = 0
            self.ie_delta_bot = 0

        return ie_prices
