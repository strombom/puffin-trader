
import pickle
from datetime import datetime


def time_mod(time, delta, epoch=None):
    if epoch is None:
        epoch = datetime(1970, 1, 1, tzinfo=time.tzinfo)
    return (time - epoch) % delta


def time_round(time, delta, epoch=None):
    mod = time_mod(time, delta, epoch)
    if mod < (delta / 2):
        return time - mod
    return time + (delta - mod)


class OrderBook:
    def __init__(self, timestamp, ask, bid):
        self.timestamp = timestamp
        self.ask = ask
        self.bid = bid
        self.mid = (ask + bid) / 2

    def __repr__(self):
        return f'OrderBook({self.timestamp} {self.ask}, {self.bid})'


def order_books_to_csv(order_books, filename):
    with open(filename, 'w') as f:
        for order_book in order_books:
            f.write(f'{int(datetime.timestamp(order_book.timestamp))},{order_book.ask},{order_book.bid}\n')


def make_order_books(agg_ticks, interval):
    try:
        with open(f"../Coastline/cache/order_books.pickle", 'rb') as f:
            data = pickle.load(f)
            return data
    except:
        pass

    if agg_ticks is None or interval is None:
        return None

    next_timestamp = time_round(agg_ticks[0].timestamp, interval) + interval

    order_books = []
    ask, bid = agg_ticks[0].high, agg_ticks[0].low
    ask_max, bid_min = ask, bid
    for agg_tick in agg_ticks:
        if agg_tick.high > ask:
            ask = agg_tick.high
        elif agg_tick.high < ask:
            ask = agg_tick.high + 0.5
        if agg_tick.low < bid:
            bid = agg_tick.low
        elif agg_tick.low > bid:
            bid = agg_tick.low - 0.5

        if agg_tick.timestamp >= next_timestamp:
            order_book = OrderBook(next_timestamp - interval, ask=ask_max, bid=bid_min)
            order_books.append(order_book)
            ask_max, bid_min = ask, bid
            next_timestamp += interval

        ask_max = max(ask_max, ask)
        bid_min = min(bid_min, bid)

    with open(f"../Coastline/cache/order_books.pickle", 'wb') as f:
        data = order_books
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return order_books
