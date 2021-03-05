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
from slopes import Slopes, make_slope
from position_live import PositionLive
from Common.Misc import PositionDirection
from IntrinsicTime.live_runner import LiveRunner


def get_historic_data(runner: LiveRunner, ie_prices: deque, initial_timestamp: datetime):
    logging.info("Connect to Binance delta server")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://192.168.1.90:31003")

    timestamp_request = initial_timestamp
    while datetime.utcnow() - timestamp_request > timedelta(seconds=2):
        command = {'command': 'get_ticks',
                   'symbol': 'BTCUSDT',
                   'max_rows': 1000,
                   'timestamp_start': timestamp_request.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'}
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
            timestamp_request = timestamp + timedelta(milliseconds=1)

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
    delta = 0.0025
    initial_timestamp = datetime.utcnow() - timedelta(hours=3)
    runner = LiveRunner(delta=delta, initial_timestamp=initial_timestamp)

    logging.info("Start Binance account")
    binance_account = BinanceAccount(binance_client)
    if binance_account.calculate_leverage() > 0:
        direction = PositionDirection.long
    else:
        direction = PositionDirection.short
    position = PositionLive(delta=delta, direction=direction)
    ie_prices = deque(maxlen=Slopes.max_slope_length + 10)

    logging.info("Get historic data")
    get_historic_data(runner=runner, ie_prices=ie_prices, initial_timestamp=initial_timestamp)
    if len(ie_prices) < ie_prices.maxlen:
        logging.error(f"Tick history not filled {len(ie_prices)}/{ie_prices.maxlen}")
        quit()

    def process_ticker_message(data):
        # {'e': 'trade', 'E': 1614958138067, 's': 'BTCUSDT', 't': 686038737, 'p': '48253.49000000', 'q': '0.00270700', 'b': 5108573132, 'a': 5108573234, 'T': 1614958138066, 'm': True, 'M': True}
        symbol = data['s']
        timestamp = datetime.utcfromtimestamp(data['E'] / 1000)
        price = float(data['p'])
        volume = float(data['q'])
        buy = not data['m']
        # trade_id = data['t']
        # print(f'sym({symbol}) ts({timestamp}) p({price}) v({volume}) buy({buy})')

        if symbol != 'BTCUSDT':
            return

        new_ie_prices = runner.step(timestamp=timestamp, price=price, volume=volume, buy=buy)
        for ie_price, ie_duration in new_ie_prices:
            ie_prices.append(ie_price)
            print(ie_prices)
            print(np.array(ie_prices))
            slope_prices = np.array(ie_prices)[-Slopes.max_slope_length - 1:-1]
            print(slope_prices)
            if slope_prices.shape[0] != Slopes.max_slope_length:
                continue

            slope = make_slope(slope_prices, Slopes.min_slope_length, Slopes.max_slope_length)
            make_trade = position.step(mark_price=ie_price, duration=ie_duration, slope=slope)
            if make_trade:
                if position.direction == PositionDirection.long:
                    position.direction = PositionDirection.short
                    logging.warning("Order -1.5!")
                    # binance_account.order(-1.5)
                else:
                    position.direction = PositionDirection.long
                    logging.warning("Order 2.5!")
                    # binance_account.order(2.5)

            print(datetime.utcnow(), ie_prices[len(ie_prices) - 1], slope['length'], slope['angle'], position.direction)

    logging.info("Start Binance BTCUSDT ticker stream")
    trade_socket_manager = BinanceSocketManager(binance_client)
    trade_socket_manager.start_trade_socket('BTCUSDT', process_ticker_message)
    trade_socket_manager.start()

    while True:
        sleep(1)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.Formatter.converter = time.gmtime
    trader()
