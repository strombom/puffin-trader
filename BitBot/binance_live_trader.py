import time

import zmq
import json
import msgpack
import logging
import numpy as np
from time import sleep
from collections import deque
from binance.client import Client
from datetime import datetime, timedelta

from binance.websockets import BinanceSocketManager
from binance_account import BinanceAccount
from slopes import Slopes
from position_live import PositionLive
from Common.Misc import PositionDirection
from IntrinsicTime.live_runner import LiveRunner


def get_historic_data(runner, ie_prices):
    logging.info("Connect to Binance delta server")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://192.168.1.90:31003")

    timestamp_start = datetime.utcnow() - timedelta(minutes=180)
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
            volume = row[2]
            buy = not row[3]
            # print(timestamp, price, buy)
            historic_data.append((timestamp, price, volume, buy))
            timestamp_start = timestamp + timedelta(milliseconds=1)

        for timestamp, price, volume, buy in historic_data:
            new_ie_prices = runner.step(timestamp=timestamp, price=price, volume=volume, buy=buy)
            ie_prices.extend(new_ie_prices)


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

    logging.info("Start Binance client")
    binance_client = Client(api_key, api_secret)

    logging.info("Start live runner")
    delta = 0.0015
    initial_price = 40000.0
    runner = LiveRunner(delta=delta, initial_price=initial_price)

    logging.info("Start Binance account")
    binance_account = BinanceAccount(binance_client)
    if binance_account.calculate_leverage() > 0:
        direction = PositionDirection.long
    else:
        direction = PositionDirection.short
    position = PositionLive(delta=delta, initial_price=initial_price, direction=direction)
    ie_prices = deque(maxlen=Slopes.max_slope_length + 10)

    logging.info("Get historic data")
    get_historic_data(runner, ie_prices)

    def process_ticker_message(data):
        # {'e': 'trade', 'E': 1614958138067, 's': 'BTCUSDT', 't': 686038737, 'p': '48253.49000000', 'q': '0.00270700', 'b': 5108573132, 'a': 5108573234, 'T': 1614958138066, 'm': True, 'M': True}
        symbol = data['s']
        timestamp = data['E']
        price = data['p']
        volume = data['q']
        buy = not data['m']
        # trade_id = data['t']
        # print(f'sym({symbol}) ts({timestamp}) p({price}) v({volume}) buy({buy}) tid({trade_id})')

        if symbol != 'BTCUSDT':
            return

        # Todo, ie_prices should be a deque with max length

        new_ie_prices = runner.step(timestamp=timestamp, price=price, volume=volume, buy=buy)
        for ie_price, ie_duration in new_ie_prices:
            ie_prices.append(ie_price)
            slope_prices = np.array(ie_prices)[-Slopes.max_slope_length - 1:-1]
            if slope_prices.shape[0] != Slopes.max_slope_length:
                continue

            slope = Slope(prices=slope_prices)
            make_trade = position.step(mark_price=ie_price, duration=ie_duration, slope=slope)
            if make_trade:
                if position.direction == PositionDirection.long:
                    position.direction = PositionDirection.short
                    binance_account.order(-1.5)
                else:
                    position.direction = PositionDirection.long
                    binance_account.order(2.5)

            print(datetime.utcnow(), ie_prices[len(ie_prices) - 1], slope.length, slope.angle, position.direction)

    logging.info("Start Binance BTCUSDT ticker stream")
    trade_socket_manager = BinanceSocketManager(binance_client)
    trade_socket_manager.start_trade_socket('BTCUSDT', process_ticker_message)
    trade_socket_manager.start()

    # margin_socket_manager = BinanceSocketManager(binance_client)
    # margin_socket_manager.start_depth_socket('BTCUSDT', process_depth_message,
    #                                          depth=BinanceSocketManager.WEBSOCKET_DEPTH_5)
    # margin_socket_manager.start()


    while True:
        sleep(1)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.Formatter.converter = time.gmtime
    trader()
