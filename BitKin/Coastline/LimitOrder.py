
from Common import OrderSide


class LimitOrder:
    def __init__(self, side, price, volume, event_type):
        self.side = side
        self.price = round(price * 2) / 2
        self.volume = volume
        self.event_type = event_type
        self.balanced_orders = []
        print(f'New order side({side}) price({self.price}) vol({volume}) event({event_type})')

    def set_balanced_orders(self, balanced_orders):
        balanced_volume = 0.0
        for order in balanced_orders:
            balanced_volume += order.volume
        self.volume = balanced_volume
        self.balanced_orders = balanced_orders

    def get_relative_pnl(self):
        pnl = 0.0
        for order in self.balanced_orders:
            if self.side == OrderSide.long:
                price_move = order.price - self.price
            else:
                price_move = self.price - order.price
            pnl += price_move / order.price * order.volume
        return pnl
