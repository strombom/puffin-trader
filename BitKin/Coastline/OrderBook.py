

class OrderBook:
    def __init__(self, ask, bid):
        self.ask = ask
        self.bid = bid
        self.mid = (ask + bid) / 2

    def __repr__(self):
        return f'OrderBook({self.ask}, {self.bid})'


def make_order_books(agg_ticks):
    order_books = []
    ask, bid = agg_ticks[0].high, agg_ticks[0].low
    for agg_tick in agg_ticks:
        order_book = OrderBook(ask=ask, bid=bid)
        order_books.append(order_book)

    return order_books
