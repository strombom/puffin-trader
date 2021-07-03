from enum import IntEnum


class Direction(IntEnum):
    up = 1
    down = -1


class Runner:
    def __init__(self, delta, initial_price, initial_timestamp):
        self.delta = delta
        self.current_price = initial_price
        self.ie_start_price = initial_price
        self.ie_max_price = initial_price
        self.ie_min_price = initial_price
        self.ie_timestamp = initial_timestamp
        self.ie_volume = 0
        self.ie_trade_count = 0
        self.ie_delta_top = 0
        self.ie_delta_bot = 0

        self.ie_times = []
        self.ie_prices = []

    def step(self, timestamp, ask, bid):







class Runner:
    def __init__(self, delta, order_book):
        self.delta = delta
        self.direction = Direction.up
        self.extreme_price = order_book.ask
        self.extreme_timestamp = order_book.timestamp
        self.expected_ie_price = order_book.ask * (1 + self.delta)
        self.os_times = []
        self.os_prices = []
        self.ie_times = []
        self.ie_prices = []

    def step(self, timestamp, ask, bid):
        if self.direction == Direction.up:
            while bid > self.expected_ie_price:
                self.ie_times.append(timestamp.timestamp())
                self.ie_prices.append(self.expected_ie_price)
                self.expected_ie_price *= 1 + self.delta

            if bid > self.extreme_price:
                self.extreme_price = order_book.bid
                self.extreme_timestamp = timestamp

            delta_dc = (self.expected_ie_price - self.extreme_price) / self.extreme_price
            expected_dc_price = self.extreme_price * (1 - delta_dc)

            if ask < expected_dc_price:
                self._append(timestamp, expected_dc_price)
                self.direction = Direction.down
                self.expected_ie_price = expected_dc_price * (1 - self.delta)
                self.extreme_timestamp = timestamp
                self.extreme_price = ask

        else:
            while ask < self.expected_ie_price:
                self.ie_times.append(timestamp.timestamp())
                self.ie_prices.append(self.expected_ie_price)
                self.expected_ie_price *= 1 - self.delta

            if ask < self.extreme_price:
                self.extreme_price = ask
                self.extreme_timestamp = timestamp

            delta_dc = (self.extreme_price - self.expected_ie_price) / self.extreme_price
            expected_dc_price = self.extreme_price * (1 + delta_dc)

            if bid > expected_dc_price:
                self._append(timestamp, expected_dc_price)
                self.direction = Direction.up
                self.expected_ie_price = expected_dc_price * (1 + self.delta)
                self.extreme_timestamp = timestamp
                self.extreme_price = bid

    def _append(self, dc_timestamp, delta_price):
        self.ie_times.append(dc_timestamp.timestamp())
        self.ie_prices.append(delta_price)
        self.os_times.append(self.extreme_timestamp.timestamp())
        self.os_prices.append(self.extreme_price)
