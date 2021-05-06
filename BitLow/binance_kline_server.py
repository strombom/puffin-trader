import itertools

import zmq
import json
import threading
from time import sleep
from datetime import datetime, timedelta
from collections import deque
from binance.client import Client
from binance.websockets import BinanceSocketManager


class BinanceKlineAccount:
    def __init__(self, binance_client, quote_asset):
        self.mark_prices = {}

        self._client = binance_client
        self._assets = {}
        self._kline_threads = {}

        trade_symbols = []
        exchange_info = self._client.get_exchange_info()
        for symbol in exchange_info['symbols']:
            if symbol['quoteAsset'] == quote_asset:
                trade_symbols.append(symbol['symbol'])

        def process_kline_message(data):
            if data['e'] == 'kline':
                self.mark_prices[data['s']] = float(data['k']['c'])

        for symbol in trade_symbols:
            kline_socket_manager = BinanceSocketManager(self._client)
            kline_socket_manager.start_kline_socket(symbol=symbol, callback=process_kline_message, interval='1m')
            kline_socket_manager.start()
            self._kline_threads[symbol] = kline_socket_manager

        from time import sleep
        has_all_symbols = False
        while not has_all_symbols:
            has_all_symbols = True
            for symbol in trade_symbols:
                if symbol not in self.mark_prices:
                    has_all_symbols = False
                    sleep(0.2)
                    break


def server():
    with open('binance_account.json') as f:
        account_info = json.load(f)
        api_key = account_info['api_key']
        api_secret = account_info['api_secret']

    binance_client = Client(api_key, api_secret)
    binance_account = BinanceKlineAccount(binance_client=binance_client, quote_asset='USDT')

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:31008")

    history = deque(maxlen=7 * 24 * 60)
    history_counter = 0

    history_lock = threading.Lock()

    def history_thread():
        next_timestamp = datetime.now()
        while True:
            next_timestamp += timedelta(minutes=1)
            with history_lock:
                nonlocal history_counter
                history_counter = history_counter + 1
                history.append(binance_account.mark_prices.copy())
                print(f"{datetime.now()} Append: {binance_account.mark_prices}")
            while next_timestamp > datetime.now():
                sleep(1.0)

    x = threading.Thread(target=history_thread)
    x.start()

    while True:
        command, payload = socket.recv_pyobj()
        send_data = None

        with history_lock:
            if command == 'get_all':
                send_data = {
                    'last_idx': history_counter,
                    'mark_prices': history
                }

            elif command == 'get_since':
                last_idx = payload
                history_idx = len(history) - 1 - (history_counter - last_idx)
                history_idx = max(0, history_idx)
                send_data = {
                    'last_idx': history_counter,
                    'mark_prices': list(itertools.islice(history, history_idx, len(history)))
                }

        socket.send_pyobj(send_data)


if __name__ == "__main__":
    server()
