from enum import IntEnum


class Direction(IntEnum):
    up = 1
    down = -1


class RunnerEvent(IntEnum):
    change_up = 1
    nothing = 0
    change_down = -1


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

    def step(self, order_book):
        if self.direction == Direction.up:
            while order_book.bid > self.expected_ie_price:
                self.ie_times.append(order_book.timestamp.timestamp())
                self.ie_prices.append(self.expected_ie_price)
                self.expected_ie_price *= 1 + self.delta

            if order_book.bid > self.extreme_price:
                self.extreme_price = order_book.bid
                self.extreme_timestamp = order_book.timestamp

            delta_dc = (self.expected_ie_price - self.extreme_price) / self.extreme_price
            expected_dc_price = self.extreme_price * (1 - delta_dc)

            if order_book.ask < expected_dc_price:
                self._append(order_book.timestamp, expected_dc_price)
                self.direction = Direction.down
                self.expected_ie_price = expected_dc_price * (1 - self.delta)
                self.extreme_timestamp = order_book.timestamp
                self.extreme_price = order_book.ask

                while order_book.ask < self.expected_ie_price:
                    self.ie_times.append(order_book.timestamp.timestamp())
                    self.ie_prices.append(self.expected_ie_price)
                    self.expected_ie_price *= 1 - self.delta

        else:
            while order_book.ask < self.expected_ie_price:
                self.ie_times.append(order_book.timestamp.timestamp())
                self.ie_prices.append(self.expected_ie_price)
                self.expected_ie_price *= 1 - self.delta

            if order_book.ask < self.extreme_price:
                self.extreme_price = order_book.ask
                self.extreme_timestamp = order_book.timestamp

            delta_dc = (self.extreme_price - self.expected_ie_price) / self.extreme_price
            expected_dc_price = self.extreme_price * (1 + delta_dc)

            if order_book.bid > expected_dc_price:
                self._append(order_book.timestamp, expected_dc_price)
                self.direction = Direction.up
                self.expected_ie_price = expected_dc_price * (1 + self.delta)
                self.extreme_timestamp = order_book.timestamp
                self.extreme_price = order_book.bid

                while order_book.bid > self.expected_ie_price:
                    self.ie_times.append(order_book.timestamp.timestamp())
                    self.ie_prices.append(self.expected_ie_price)
                    self.expected_ie_price *= 1 + self.delta

    def _append(self, dc_timestamp, delta_price):
        self.ie_times.append(dc_timestamp.timestamp())
        self.ie_prices.append(delta_price)
        self.os_times.append(self.extreme_timestamp.timestamp())
        self.os_prices.append(self.extreme_price)
