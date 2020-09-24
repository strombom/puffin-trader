
import csv


filename = "../events.csv"


maker_fee = -0.00025
taker_fee = 0.00075
max_leverage = 10.0


class Event:
    def __init__(self, timestamp, price, price_min, price_max, price_execute):
        self.timestamp = timestamp
        self.price = price
        self.price_execute = price_execute

    def __repr__(self):
        return f"Event({self.timestamp}, {self.price}, {self.price_execute})"


events = []
with open(filename) as csv_file:
    for row in csv.reader(csv_file):
        events.append(Event(row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4])))


class Position:
    def __init__(self, mark_price, initial_leverage, stop_loss, min_profit):
        self.wallet = 1.0
        self.price = mark_price
        self.contracts = initial_leverage * self.wallet * self.price
        self.stop_loss = stop_loss
        self.min_profit = min_profit
        self._update()

    def _update(self):
        self.direction = self.contracts / abs(self.contracts)
        self.min_profit_price = self.price * (1 + self.direction * min_profit)
        self.stop_loss_price = self.price * (1 - self.direction * stop_loss)

    def market_order(self, order_leverage, mark_price):
        # print('market_order', wallet, pos_price, pos_contracts, order_contracts, mark_price)

        sign = self.contracts / abs(self.contracts)
        entry_value = abs(self.contracts / position.price)
        mark_value = abs(self.contracts / mark_price)
        upnl = sign * (mark_value - entry_value)

        max_contracts = max_leverage * (self.wallet + upnl) * mark_price
        margin = self.wallet * min(max(order_leverage, -max_leverage), max_leverage)
        order_contracts = min(max(margin * mark_price, -max_contracts), max_contracts) - self.contracts

        # Fee
        self.wallet -= taker_fee * abs(order_contracts / mark_price)

        # Realised profit and loss
        # Wallet only changes when abs(contracts) decrease
        if (self.contracts > 0) and (order_contracts < 0):
            self.wallet += (1 / self.price - 1 / mark_price) * min(-order_contracts, self.contracts)
        elif (self.contracts < 0) and (order_contracts > 0):
            self.wallet += (1 / self.price - 1 / mark_price) * max(-order_contracts, self.contracts)

        # Calculate average entry price
        if (self.contracts >= 0 and order_contracts > 0) or (self.contracts <= 0 and order_contracts < 0):
            self.price = (self.contracts * self.price + order_contracts * mark_price) / (self.contracts + order_contracts)
        elif (self.contracts >= 0 and self.contracts + order_contracts < 0) or (self.contracts <= 0 and self.contracts + order_contracts > 0):
            self.price = mark_price

        # Calculate position contracts
        self.contracts += order_contracts
        self._update()


for stop_loss in [0.04, 0.06, 0.10]:
    for min_profit in [0.005, 0.01, 0.015]:

        position = Position(events[0].price, max_leverage, stop_loss, min_profit)
        for event in events:
            if position.direction > 0:
                if event.price < position.stop_loss_price or event.price > position.min_profit_price:
                    position.market_order(-max_leverage, event.price)
                    #print(f"go short, value({position.wallet * event.price} price({event.price})")

            if position.direction < 0:
                if event.price > position.stop_loss_price or event.price < position.min_profit_price:
                    position.market_order(max_leverage, event.price)
                    #print(f"go long, value({position.wallet * event.price} price({event.price})")

        print(f"--- stop_loss({stop_loss}) min_profit({min_profit})")
        print(f"value({position.wallet * events[-1].price:.2f}) wallet({position.wallet:.2f}) price({events[-1].price:.2f})")
