
import logging


class LiveRunner:
    price_change_volume_threshold = 10.0

    def __init__(self, delta, initial_price):
        self.delta = delta
        self.current_price = initial_price
        self.previous_price = initial_price
        self.ie_timestamp = 0
        self.ie_volume = 0
        self.ie_start_price = initial_price
        self.ie_max_price = initial_price
        self.ie_min_price = initial_price
        self.ie_delta_top = 0
        self.ie_delta_bot = 0
        self.ie_trade_count = 0

        self.buy_accum_volume = 0
        self.sell_accum_volume = 0

    def step(self, timestamp: int, price: float, volume: float, buy: bool):
        events = []

        if buy:
            self.buy_accum_volume += volume
            if self.buy_accum_volume > self.price_change_volume_threshold:
                self.buy_accum_volume = 0
                self.current_price = min(self.current_price, price)
        else:
            self.sell_accum_volume += volume
            if self.sell_accum_volume > self.price_change_volume_threshold:
                self.sell_accum_volume = 0
                self.current_price = max(self.current_price, price)

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

        self.ie_trade_count += 1

        delta_down = (self.ie_max_price - self.current_price) / self.ie_max_price  # Delta from top
        delta_up = (self.current_price - self.ie_min_price) / self.ie_min_price    # Delta from bot

        if self.ie_delta_top + delta_down >= self.delta or self.ie_delta_bot + delta_up >= self.delta:
            ie_duration = timestamp - self.ie_timestamp

            if delta_dir == 1:
                remaining_delta = self.ie_delta_bot + delta_up
                ie_price = self.ie_min_price * (1.0 + (self.delta - self.ie_delta_bot))
            else:
                remaining_delta = self.ie_delta_top + delta_down
                ie_price = self.ie_max_price * (1.0 + (self.delta - self.ie_delta_top))

            # ie_delta = (self.ie_start_price - ie_price) / self.ie_start_price

            while remaining_delta >= 2 * self.delta:
                if delta_dir == 1:
                    self.ie_max_price = min(self.ie_max_price, ie_price)
                else:
                    self.ie_min_price = max(self.ie_min_price, ie_price)

                events.append((ie_price, ie_duration))

                next_price = ie_price * (1.0 + delta_dir * self.delta)
                self.ie_start_price = ie_price
                self.ie_volume = 0
                self.ie_trade_count = 0
                if delta_dir == 1:
                    self.ie_max_price = next_price
                    self.ie_min_price = ie_price
                else:
                    self.ie_max_price = ie_price
                    self.ie_min_price = next_price
                self.ie_delta_top = (self.ie_max_price - self.ie_start_price) / self.ie_start_price
                self.ie_delta_bot = (self.ie_start_price - self.ie_min_price) / self.ie_start_price
                # ie_delta = (self.ie_start_price - next_price) / self.ie_start_price
                ie_price = next_price

            self.ie_volume += volume
            events.append((ie_price, ie_duration))

            self.ie_timestamp = timestamp
            self.ie_start_price = ie_price
            self.ie_volume = 0
            self.ie_trade_count = 0
            self.ie_max_price = ie_price
            self.ie_min_price = ie_price
            self.ie_delta_top = 0
            self.ie_delta_bot = 0

        else:
            self.ie_volume += volume

        """
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
        """
