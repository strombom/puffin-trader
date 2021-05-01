from copy import copy
from datetime import datetime

import zmq
import json
import time
import queue
import logging
import threading
import numpy as np
from collections import deque

from IntrinsicTime.runner import Runner
from binance_account import BinanceAccount


class Indicators:
    def __init__(self, config: dict):
        self._config = config
        self._data = {}
        self._directions = {}
        self._indicator_queue = queue.Queue()
        self._runners = {}
        self._running = True
        self._updater_thread = threading.Thread(target=self._updater)
        self._updater_thread.start()
        self._initialized = False

    def get_pairs(self):
        return list(self._directions.keys())

    def get_queue(self):
        return self._indicator_queue

    def is_initialized(self):
        return self._initialized

    def _calculate_indicators(self):
        # for pair in self._directions:
        #     print(pair)
        directions = dict(sorted(self._directions.items(), reverse=False, key=lambda item: item[1]))
        directions = {key: val for key, val in directions.items() if val != -1.0}
        top_pairs = set(list(directions.keys())[:self._config['portfolio_size']])
        self._indicator_queue.put(top_pairs)

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

        self._calculate_indicators()

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
                        self._data[pair].append(ie_price)
                        updated = True

            if updated:
                self._calculate_directions()
                self._initialized = True

            time.sleep(1.0)


def trader():
    with open('binance_account.json') as f:
        account_info = json.load(f)
        api_key = account_info['api_key']
        api_secret = account_info['api_secret']

    with open('config.json') as f:
        config = json.load(f)

    indicators = Indicators(config)
    while not indicators.is_initialized():
        time.sleep(0.5)

    logging.info("Start Binance client")
    binance_account = BinanceAccount(api_key, api_secret, trade_pairs=indicators.get_pairs())

    indicator_queue = indicators.get_queue()
    while True:
        top_pairs = indicator_queue.get()
        print(f"Indicator update: {datetime.now()}")
        print(f"Top pairs {top_pairs}")

        portfolio = binance_account.get_portfolio()
        print(f"Portfolio {portfolio}")
        for portfolio_pair in portfolio.copy():
            if portfolio_pair not in top_pairs:
                balance = binance_account.get_balance(portfolio_pair)
                print(f"Sell {balance} {portfolio_pair}")
                binance_account.market_sell(trade_pair=portfolio_pair, volume=balance)
                portfolio.remove(portfolio_pair)

        portfolio = binance_account.get_portfolio()
        for top_pair in top_pairs:
            if top_pair not in portfolio:
                total_equity = binance_account.get_total_equity_usdt()
                mark_price = binance_account.get_mark_price(top_pair)
                max_volume = binance_account.get_balance_usdt() / mark_price
                volume = total_equity / (config['portfolio_size'] * mark_price)
                volume = min(volume, max_volume) * 0.95
                print(f"Buy {volume} {top_pair}")
                binance_account.market_buy(trade_pair=top_pair, volume=volume)
                portfolio = binance_account.get_portfolio()


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.Formatter.converter = time.gmtime

    while True:
        try:
            trader()
        except ConnectionError as e:
            print("Error!", e)
            time.sleep(10)
