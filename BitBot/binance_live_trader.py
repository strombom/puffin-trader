
import zmq
import json
import signal
import msgpack
import threading
import ratelimit
import numpy as np
from collections import deque
from multiprocessing import Pipe
from datetime import datetime, timedelta

from binance_account import BinanceAccount
from slopes import Slopes, Slope
from position_live import PositionLive
from Common.Misc import PositionDirection
from IntrinsicTime.live_runner import LiveRunner


def live_data(data_pipe: Pipe):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://192.168.1.90:31003")

    def get_data(last_timestamp: datetime):
        timestamp_start_str = last_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'
        command = {'command': 'get_ticks',
                   'symbol': 'BTCUSDT',
                   'max_rows': 1000,
                   'timestamp_start': timestamp_start_str}
        # print(command)
        command = json.dumps(command).encode()
        socket.send(command)

        message = socket.recv()
        message = msgpack.unpackb(message)

        for row in message:
            # print(row[0] // 1000)
            timestamp = datetime.utcfromtimestamp(row[0] / 1000)
            price = row[1]
            buy = not row[3]
            # print(timestamp, price, buy)
            data_pipe.send((timestamp, price, buy))
            last_timestamp = timestamp + timedelta(milliseconds=1)

        return last_timestamp

    timestamp_start = datetime.utcnow() - timedelta(minutes=2)
    while True:
        if data_pipe.poll(0.5):
            print("live_data has msg " + data_pipe.recv())
            data_pipe.send({'command': 'quit'})
            break
        # print("t start", timestamp_start)
        timestamp_start = get_data(timestamp_start)
        # print("t new", timestamp_start)


class LivePlotter:
    def __init__(self):
        pass

    def append_threshold(self, ie_idx, threshold):
        pass

    def regime_change(self, x, mark_price, regime):
        pass


def trader(data_pipe: Pipe):
    binance_account = BinanceAccount()
    from time import sleep
    while True:
        sleep(1)

    initial_price = 40000.0
    runner = LiveRunner(delta=0.001, initial_price=initial_price)

    @ratelimit.limits(calls=1, period=1)
    def print_it(msg):
        # print("-> " + msg)
        pass

    position = PositionLive(delta=0.001, initial_price=initial_price, direction=PositionDirection.long)
    ie_prices = deque(maxlen=Slopes.max_slope_length + 10)
    ask, bid = initial_price, initial_price

    while True:
        if data_pipe.poll(0.5):
            timestamp, price, buy = data_pipe.recv()
            if buy:
                ask = price
            else:
                bid = price

            make_trade = None
            new_ie_prices = runner.step(ask=ask, bid=bid)
            if len(new_ie_prices) > 0:
                for ie_price in new_ie_prices:
                    ie_prices.append(ie_price)
                    slope_prices = np.array(ie_prices)[-Slopes.max_slope_length - 1:-1]
                    if slope_prices.shape[0] == Slopes.max_slope_length:
                        slope = Slope(prices=slope_prices)
                        make_trade = position.step(mark_price=ie_price, slope=slope)
                        if make_trade:
                            if position.direction == PositionDirection.long:
                                position.direction = PositionDirection.short
                                binance_account.order(-1)
                            else:
                                position.direction = PositionDirection.long
                                binance_account.order(1)

                        print(timestamp, ie_prices[len(ie_prices) - 1], slope.length, slope.angle, position.direction)

            try:
                print_it(f"trader has msg {timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')} {ask:.1f} <-> {bid:.1f} {buy}")
            except ratelimit.exception.RateLimitException:
                pass


if __name__ == '__main__':

    live_data_pipe, trader_pipe = Pipe()

    live_data_thread = threading.Thread(target=live_data, args=(live_data_pipe, ))
    live_data_thread.start()

    trader_thread = threading.Thread(target=trader, args=(trader_pipe, ))
    trader_thread.start()

    def signal_handler(signal_, frame):
        live_data_pipe.send({'command': 'quit'})

    signal.signal(signal.SIGINT, signal_handler)

    live_data_thread.join()
    trader_thread.join()
