
import sys
sys.path.append("../Common")


from Common import OrderSide
from OrderBook import make_order_books
from misc import read_agg_ticks
from CoastlineTrader import CoastlineTrader


class Price:
    def __init__(self, ask, bid):
        self.ask = ask
        self.bid = bid
        self.mid = (ask + bid) / 2


if __name__ == '__main__':
    agg_ticks = read_agg_ticks('C:/development/github/puffin-trader/tmp/agg_ticks.csv')
    order_books = make_order_books(agg_ticks)

    trader = CoastlineTrader(0.0025, OrderSide.long)
    for order_book in order_books:
        trader.step(order_book)
