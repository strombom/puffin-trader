

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
    prev_ask, prev_bid = 0, 0
    for agg_tick in agg_ticks:
        if agg_tick.high > ask:
            ask = agg_tick.high
        elif agg_tick.high < ask:
            ask = agg_tick.high + 0.5
        if agg_tick.low < bid:
            bid = agg_tick.low
        elif agg_tick.low > bid:
            bid = agg_tick.low - 0.5

        if ask != prev_ask or bid != prev_bid:
            order_books.append(OrderBook(ask=ask, bid=bid))
            prev_ask, prev_bid = ask, bid

    return order_books
