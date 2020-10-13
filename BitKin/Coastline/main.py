
import sys
sys.path.append("../Common")

import matplotlib.pyplot as plt
from Common import OrderSide
from OrderBook import make_order_books
from misc import read_agg_ticks
from CoastlineTrader import CoastlineTrader


if __name__ == '__main__':
    agg_ticks = read_agg_ticks('C:/development/github/puffin-trader/tmp/agg_ticks.csv')
    order_books = make_order_books(agg_ticks)

    #prices = []
    #for order_book in order_books:
    #    prices.append(order_book.mid)
    #plt.plot(prices)
    #plt.show()

    trader = CoastlineTrader(0.01, OrderSide.long, order_books[0].mid)
    for order_book in order_books:
        trader.step(order_book)
