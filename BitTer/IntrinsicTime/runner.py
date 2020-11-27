from enum import Enum


class Direction(Enum):
    up = 1
    down = -1


class RunnerEvent(Enum):
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
        self.dc_times = []
        self.dc_prices = []
        self.os_times = []
        self.os_prices = []
        self.ie_times = []
        self.ie_prices = []

    def step(self, order_book):
        event = RunnerEvent.nothing, 0, 0, 0, 0

        if self.direction == Direction.up:
            while order_book.ask > self.ie_price:
                #print(f'up IE {self.ie_price}')
                self.ie_times.append(order_book.timestamp.timestamp())
                self.ie_prices.append(self.ie_price)
                self.ie_price *= 1 + self.delta_up

            if order_book.ask > self.extreme_price:
                self.extreme_price = order_book.ask
                self.extreme_timestamp = order_book.timestamp
                self.delta_price = order_book.ask * (1 - self.delta_down)

            if order_book.ask < self.delta_price:
                #print(f'up->down DC {order_book.ask}')
                self._append(order_book.timestamp)
                self.direction = Direction.down
                self.ie_price = self.delta_price * (1 - self.delta_down)
                self.extreme_timestamp = order_book.timestamp
                self.extreme_price = order_book.bid
                self.delta_price = self.extreme_price * (1 + self.delta_up)
                event = RunnerEvent.change_down

                while self.ie_price > order_book.bid:
                    # print(f'down IE {self.ie_price}')
                    self.ie_times.append(order_book.timestamp.timestamp())
                    self.ie_prices.append(self.ie_price)
                    self.ie_price *= 1 - self.delta_down

        else:
            while order_book.bid < self.ie_price:
                #print(f'down IE {self.ie_price}')
                self.ie_times.append(order_book.timestamp.timestamp())
                self.ie_prices.append(self.ie_price)
                self.ie_price *= 1 - self.delta_down

            if order_book.bid < self.extreme_price:
                self.extreme_price = order_book.bid
                self.extreme_timestamp = order_book.timestamp
                self.delta_price = order_book.bid * (1 + self.delta_up)

            if order_book.bid > self.delta_price:
                #print(f'down->up DC {order_book.ask}')
                self._append(order_book.timestamp)
                self.direction = Direction.up
                self.ie_price = self.delta_price * (1 + self.delta_up)
                self.extreme_timestamp = order_book.timestamp
                self.extreme_price = order_book.ask
                self.delta_price = self.extreme_price * (1 - self.delta_down)
                event = RunnerEvent.change_up

                while self.ie_price < order_book.ask:
                    #print(f'up IE {self.ie_price}')
                    self.ie_times.append(order_book.timestamp.timestamp())
                    self.ie_prices.append(self.ie_price)
                    self.ie_price *= 1 + self.delta_up

        return event

    def _append(self, dc_timestamp):
        self.ie_times.append(dc_timestamp.timestamp())
        self.ie_prices.append(self.delta_price)
        self.os_times.append(self.extreme_timestamp.timestamp())
        self.os_prices.append(self.extreme_price)
        self.dc_times.append(dc_timestamp.timestamp())
        self.dc_prices.append(self.delta_price)
