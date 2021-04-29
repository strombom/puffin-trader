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
    def __init__(self, binance_client, top_symbols):
        self.mark_prices = {}

        self._client = binance_client
        self._assets = {}
        self._kline_threads = {}

        all_symbols = {}
        exchange_info = self._client.get_exchange_info()
        for symbol in exchange_info['symbols']:
            if symbol['baseAsset'] not in all_symbols:
                all_symbols[symbol['baseAsset']] = []
            all_symbols[symbol['baseAsset']].append(symbol['symbol'])

        trade_symbols = []
        for top_symbol in top_symbols:
            if top_symbol in all_symbols:
                for symbol in all_symbols[top_symbol]:
                    if 'USDT' in symbol:
                        trade_symbols.append(top_symbol)
                        break

        def process_trade_message(data):
            if data['e'] == 'kline':
                self.mark_prices[data['s']] = float(data['k']['c'])

        for symbol in trade_symbols:
            trade_socket_manager = BinanceSocketManager(self._client)
            trade_socket_manager.start_kline_socket(symbol=symbol + "USDT", callback=process_trade_message, interval='1m')
            trade_socket_manager.start()
            self._kline_threads[symbol] = trade_socket_manager

        from time import sleep
        has_all_symbols = False
        while not has_all_symbols:
            has_all_symbols = True
            for symbol in trade_symbols:
                if symbol + "USDT" not in self.mark_prices:
                    has_all_symbols = False
                    sleep(0.2)
                    break


top_symbols = [
    'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOT', 'LTC', 'XLM', 'BCH', 'THETA', 'FIL', 'TRX', 'DOGE', 'VET', 'SOL', 'EOS',
    'XMR', 'LUNA', 'IOTA', 'XTZ', 'ATOM', 'NEO', 'AVAX', 'ALGO', 'EGLD', 'XEM', 'DASH', 'HBAR', 'ZEC',
    'NEAR', 'ETC', 'ZIL', 'BTG', 'TFUEL', 'WAVES', 'ICX', 'RVN', 'ONT', 'QTUM', 'ONE', 'HNT', 'SC', 'DGB',
    'FTM', 'IOST', 'CELO', 'ZEN', 'LSK', 'NANO', 'CKB', 'XVG', 'NKN', 'BCD', 'ARDR', 'IOTX', 'STEEM', 'KMD',
    'STRAX', 'SRM', 'BTS', 'ARK', 'IRIS', 'ROSE', 'SYS', 'SCRT', 'HIVE', 'TOMO', 'BTM', 'ELA', 'AION', 'CTC', 'CCXX',
    'DFI', 'XWC', 'XDC', 'BCHA', 'ETN', 'AKT', 'HTR', 'ARRR', 'CRU', 'MARO', 'EDG', 'COTI', 'HNC', 'RDD', 'META',
    'LTO', 'VRA', 'BCN', 'MWC', 'PHA', 'REV', 'KDA', 'NRG', 'APL', 'MONA', 'PAC', 'TT', 'NYE', 'FIRO', 'SAPP'
]
"""
'DCR', 'KSM', 'WAN', 'STX', 
"""


def server():
    with open('binance_account.json') as f:
        account_info = json.load(f)
        api_key = account_info['api_key']
        api_secret = account_info['api_secret']

    binance_client = Client(api_key, api_secret)
    binance_account = BinanceKlineAccount(binance_client=binance_client, top_symbols=top_symbols)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:31007")

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
