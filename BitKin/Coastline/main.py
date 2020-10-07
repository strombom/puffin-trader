
import sys
sys.path.append("../Common")


from Common import OrderSide
from CoastlineTrader import CoastlineTrader


class Price:
    def __init__(self, ask, bid):
        self.ask = ask
        self.bid = bid
        self.mid = (ask + bid) / 2


if __name__ == '__main__':
    print("hej")
    trader = CoastlineTrader(0.0025, OrderSide.long)

    prices = [Price(ask=10010, bid=10000),
              Price(ask=10030, bid=10020),
              Price(ask=10050, bid=10040)]

    for mark_price in prices:
        trader.step(mark_price)
