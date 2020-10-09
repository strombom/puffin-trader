

class LimitOrder:
    def __init__(self, side, price, volume, event_type):
        self.side = side
        self.price = price
        self.volume = volume
        self.event_type = event_type

    def balance_orders(self, orders):
        pass
