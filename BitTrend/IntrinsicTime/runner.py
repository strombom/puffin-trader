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
        self.delta_up = delta * 1.0
        self.delta_down = delta * 1.0
        self.direction = Direction.up
        self.extreme_price = order_book.ask
        self.extreme_timestamp = order_book.timestamp
        self.delta_price = order_book.ask * (1 - self.delta_down)
        self.ie_price = order_book.ask * (1 + self.delta_up)
        self.ie_min_ask = float('inf')
        self.ie_max_bid = float('-inf')
        self.prev_ask = self.ie_min_ask
        self.prev_bid = self.ie_max_bid

        self.dc_times = []
        self.dc_prices = []
        self.os_times = []
        self.os_prices = []
        self.ie_times = []
        self.ie_prices = []
        self.ie_asks = []
        self.ie_bids = []
        self.ie_min_asks = []  # Since last IE
        self.ie_max_bids = []  # Since last IE

    def step(self, order_book):
        event = RunnerEvent.nothing, 0, 0, 0, 0

        def ie_append(price):
            self.ie_times.append(order_book.timestamp.timestamp())
            self.ie_prices.append(price)
            self.ie_asks.append(order_book.ask)
            self.ie_bids.append(order_book.bid)
            self.ie_min_asks.append(self.ie_min_ask)
            self.ie_max_bids.append(self.ie_max_bid)
            self.ie_min_ask = float('inf')
            self.ie_max_bid = float('-inf')
            self.prev_ask = float('-inf')
            self.prev_bid = float('inf')

        if order_book.ask < self.prev_ask:
            self.ie_min_ask = min(self.ie_min_ask, order_book.ask)
        self.prev_ask = order_book.ask

        if order_book.bid > self.prev_bid:
            self.ie_max_bid = max(self.ie_max_bid, order_book.bid)
        self.prev_bid  = order_book.bid

        if self.direction == Direction.up:
            while order_book.bid > self.ie_price:
                #print(f'up IE {self.ie_price}')
                ie_append(self.ie_price)
                self.ie_price *= 1 + self.delta_up

            if order_book.bid > self.extreme_price:
                self.extreme_price = order_book.bid
                self.extreme_timestamp = order_book.timestamp
                self.delta_price = order_book.bid * (1 - self.delta_down)

            if order_book.ask < self.delta_price:
                #print(f'up->down DC {order_book.ask}')
                self._append(order_book.timestamp)
                ie_append(self.delta_price)
                self.direction = Direction.down
                self.ie_price = self.delta_price * (1 - self.delta_down)
                self.extreme_timestamp = order_book.timestamp
                self.extreme_price = order_book.ask
                self.delta_price = self.extreme_price * (1 + self.delta_up)
                event = RunnerEvent.change_down

                while order_book.ask < self.ie_price:
                    # print(f'down IE {self.ie_price}')
                    ie_append(self.ie_price)
                    self.ie_price *= 1 - self.delta_down

        else:
            while order_book.ask < self.ie_price:
                #print(f'down IE {self.ie_price}')
                ie_append(self.ie_price)
                self.ie_price *= 1 - self.delta_down

            if order_book.ask < self.extreme_price:
                self.extreme_price = order_book.ask
                self.extreme_timestamp = order_book.timestamp
                self.delta_price = order_book.ask * (1 + self.delta_up)

            if order_book.bid > self.delta_price:
                #print(f'down->up DC {order_book.ask}')
                self._append(order_book.timestamp)
                ie_append(self.delta_price)
                self.direction = Direction.up
                self.ie_price = self.delta_price * (1 + self.delta_up)
                self.extreme_timestamp = order_book.timestamp
                self.extreme_price = order_book.bid
                self.delta_price = self.extreme_price * (1 - self.delta_down)
                event = RunnerEvent.change_up

                while order_book.bid > self.ie_price:
                    #print(f'up IE {self.ie_price}')
                    ie_append(self.ie_price)
                    self.ie_price *= 1 + self.delta_up

        return event

    def _append(self, dc_timestamp):
        self.os_times.append(self.extreme_timestamp.timestamp())
        self.os_prices.append(self.extreme_price)
        self.dc_times.append(dc_timestamp.timestamp())
        self.dc_prices.append(self.delta_price)
