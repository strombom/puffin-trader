

class LimitOrder:
    def __init__(self, side, price, volume, event_type):
        self.side = side
        self.price = round(price * 2) / 2
        self.volume = volume
        self.event_type = event_type
        print(f'New order side({side}) price({self.price}) vol({volume}) event({event_type})')

    def balance_orders(self, orders):
        pass
