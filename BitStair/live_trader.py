import json
import threading
import time
import logging
from collections import deque

import zmq
from binance.client import Client

from IntrinsicTime.runner import Runner


class Runners:
    def __init__(self, config: dict):
        self.config = config
        self._data = {}
        self._runners = {}
        self.running = True
        self._updater_thread = threading.Thread(target=self._updater)
        self._updater_thread.start()
        self._updater_thread.join()

    def _updater(self):
        next_idx = 0
        while self.running:
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect(self.config['binance_kline_server_address'])
            socket.send_pyobj(("get_since", next_idx))
            message = socket.recv_pyobj()
            next_idx = message['last_idx'] + 1

            for prices in message['mark_prices']:
                for pair in prices:
                    if pair not in self._data:
                        self._runners[pair] = Runner(delta=self.config['delta'])
                        self._data[pair] = deque(maxlen=self.config['lengths'][-1])
                    ie_prices = self._runners[pair].step(high=prices[pair], low=prices[pair])
                    for ie_price in ie_prices:
                        print(f"Append ({pair}): {ie_price}")
                        self._data[pair].append(ie_price)

            # print(message['last_idx'], len(message['mark_prices']), len(self._data['ONEUSDT']))

            for pair in self._data:
                print(f"{len(self._data[pair])} ", end='')
            print()

            time.sleep(1.0)


def trader():
    with open('binance_account.json') as f:
        account_info = json.load(f)
        api_key = account_info['api_key']
        api_secret = account_info['api_secret']

    with open('config.json') as f:
        config = json.load(f)

    runners = Runners(config)

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
