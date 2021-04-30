import zmq
import json
import time
import logging
import threading
import numpy as np
from collections import deque
from binance.client import Client

from IntrinsicTime.runner import Runner


class Indicators:
    def __init__(self, config: dict):
        self._config = config
        self._data = {}
        self._directions = {}
        self._runners = {}
        self._running = True
        self._updater_thread = threading.Thread(target=self._updater)
        self._updater_thread.start()
        self._updater_thread.join()

    def _calculate_directions(self):
        for pair in self._data:
            self._directions[pair] = -1.0
            prices = np.array(self._data[pair])
            if prices.shape[0] != self._config['lengths'][-1]:
                continue

            for length_idx, length in enumerate(self._config['lengths']):
                start, end = prices.shape[0] - length, prices.shape[0]
                xp = np.arange(start, end)
                yp = np.poly1d(np.polyfit(xp, prices[start:end], self._config['poly_order']))
                curve = yp(xp)
                direction = curve[-1] / curve[-2] - 1.0
                self._directions[pair] = max(self._directions[pair], direction)

    def _updater(self):
        next_idx = 0

        while self._running:
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect(self._config['binance_kline_server_address'])
            socket.send_pyobj(("get_since", next_idx))
            message = socket.recv_pyobj()
            next_idx = message['last_idx'] + 1

            updated = False
            for prices in message['mark_prices']:
                for pair in prices:
                    if pair not in self._data:
                        self._runners[pair] = Runner(delta=self._config['delta'])
                        self._data[pair] = deque(maxlen=self._config['lengths'][-1])
                    ie_prices = self._runners[pair].step(high=prices[pair], low=prices[pair])
                    for ie_price in ie_prices:
                        # print(f"Append ({pair}): {ie_price}")
                        self._data[pair].append(ie_price)
                        updated = True

            # print(message['last_idx'], len(message['mark_prices']), len(self._data['ONEUSDT']))

            if updated:
                self._calculate_directions()

            # for pair in self._data:
            #     print(f"{pair}: {len(self._data[pair])}, ", end='')
            # print()

            time.sleep(1.0)


def trader():
    with open('binance_account.json') as f:
        account_info = json.load(f)
        api_key = account_info['api_key']
        api_secret = account_info['api_secret']

    with open('config.json') as f:
        config = json.load(f)

    indicators = Indicators(config)

    lengths = config['lengths']
    print(lengths)
    quit()

    logging.info("Start Binance client")
    binance_client = Client(api_key, api_secret)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.Formatter.converter = time.gmtime
    trader()
