from enum import IntEnum


class Direction(IntEnum):
    up = 1
    down = -1


class RunnerEvent(IntEnum):
    change_up = 1
    nothing = 0
    change_down = -1


class LiveRunner:
    def __init__(self, delta, initial_price):
        self.delta = delta
        self.direction = Direction.up
        self.extreme_price = initial_price
        self.expected_ie_price = initial_price * (1 + self.delta)

    def step(self, ask, bid):
        ie_prices = []

        if self.direction == Direction.up:
            while bid > self.expected_ie_price:
                ie_prices.append(self.expected_ie_price)
                self.expected_ie_price *= 1 + self.delta

            if bid > self.extreme_price:
                self.extreme_price = bid

            delta_dc = (self.expected_ie_price - self.extreme_price) / self.extreme_price
            expected_dc_price = self.extreme_price * (1 - delta_dc)

            if ask < expected_dc_price:
                ie_prices.append(expected_dc_price)
                self.direction = Direction.down
                self.expected_ie_price = expected_dc_price * (1 - self.delta)
                self.extreme_price = ask

        else:
            while ask < self.expected_ie_price:
                ie_prices.append(self.expected_ie_price)
                self.expected_ie_price *= 1 - self.delta

            if ask < self.extreme_price:
                self.extreme_price = ask

            delta_dc = (self.extreme_price - self.expected_ie_price) / self.extreme_price
            expected_dc_price = self.extreme_price * (1 + delta_dc)

            if bid > expected_dc_price:
                ie_prices.append(expected_dc_price)
                self.direction = Direction.up
                self.expected_ie_price = expected_dc_price * (1 + self.delta)
                self.extreme_price = bid

        return ie_prices
