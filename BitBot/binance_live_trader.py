import zmq
import json
import msgpack
import numpy as np
from time import sleep
from collections import deque
from binance.client import Client
from datetime import datetime, timedelta

from binance.websockets import BinanceSocketManager
from binance_account import BinanceAccount
from slopes import Slopes, Slope
from position_live import PositionLive
from Common.Misc import PositionDirection
from IntrinsicTime.live_runner import LiveRunner


def get_historic_data(runner, ie_prices):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://192.168.1.90:31003")

    timestamp_start = datetime.utcnow() - timedelta(minutes=60)
    while datetime.utcnow() - timestamp_start > timedelta(seconds=2):
        command = {'command': 'get_ticks',
                   'symbol': 'BTCUSDT',
                   'max_rows': 1000,
                   'timestamp_start': timestamp_start.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'}
        command = json.dumps(command).encode()
        socket.send(command)
        message = msgpack.unpackb(socket.recv())

        historic_data = []
        for row in message:
            # print(row[0] // 1000)
            timestamp = datetime.utcfromtimestamp(row[0] / 1000)
            price = row[1]
            buy = not row[3]
            # print(timestamp, price, buy)
            historic_data.append((timestamp, price, buy))
            timestamp_start = timestamp + timedelta(milliseconds=1)

        ask, bid = 0, 0
        for timestamp, price, buy in historic_data:
            if buy:
                ask = price
            else:
                bid = price
            if ask == 0:
                ask = bid
            if bid == 0:
                bid = ask
            runner.step(ask=ask, bid=bid)
            new_ie_prices = runner.step(ask=ask, bid=bid)
            if len(new_ie_prices) > 0:
                for ie_price in new_ie_prices:
                    ie_prices.append(ie_price)


class LivePlotter:
    def __init__(self):
        pass

    def append_threshold(self, ie_idx, threshold):
        pass

    def regime_change(self, x, mark_price, regime):
        pass


def trader():
    with open('binance_account.json') as f:
        account_info = json.load(f)
        api_key = account_info['api_key']
        api_secret = account_info['api_secret']

    binance_client = Client(api_key, api_secret)

    delta = 0.0015
    initial_price = 40000.0
    runner = LiveRunner(delta=delta, initial_price=initial_price)

    binance_account = BinanceAccount(binance_client)
    position = PositionLive(delta=delta, initial_price=initial_price, direction=PositionDirection.long)
    ie_prices = deque(maxlen=Slopes.max_slope_length + 10)
    get_historic_data(runner, ie_prices)

    def process_depth_message(data):
        ask, bid = float(data['asks'][0][0]), float(data['bids'][0][0])
        new_ie_prices = runner.step(ask=ask, bid=bid)
        if len(new_ie_prices) > 0:
            for ie_price in new_ie_prices:
                ie_prices.append(ie_price)
                slope_prices = np.array(ie_prices)[-Slopes.max_slope_length - 1:-1]
                if slope_prices.shape[0] != Slopes.max_slope_length:
                    continue

                slope = Slope(prices=slope_prices)
                make_trade = position.step(mark_price=ie_price, slope=slope)
                if make_trade:
                    if position.direction == PositionDirection.long:
                        position.direction = PositionDirection.short
                        binance_account.order(-1.5)
                    else:
                        position.direction = PositionDirection.long
                        binance_account.order(2.5)

                print(datetime.utcnow(), ie_prices[len(ie_prices) - 1], slope.length, slope.angle, position.direction)

    margin_socket_manager = BinanceSocketManager(binance_client)
    margin_socket_manager.start_depth_socket('BTCUSDT', process_depth_message,
                                             depth=BinanceSocketManager.WEBSOCKET_DEPTH_5)
    margin_socket_manager.start()

    while True:
        sleep(1)


if __name__ == '__main__':
    trader()
