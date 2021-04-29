import itertools

import zmq
import json
import threading
from time import sleep
from datetime import datetime, timedelta
from collections import deque
from binance.client import Client

from binance_account import BinanceAccount


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

top_symbols = [
    'BTC', 'ETH', 'BNB'
]

def server():
    with open('binance_account.json') as f:
        account_info = json.load(f)
        api_key = account_info['api_key']
        api_secret = account_info['api_secret']

    binance_client = Client(api_key, api_secret)
    binance_account = BinanceAccount(binance_client=binance_client, top_symbols=top_symbols)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:31007")

    history = deque(maxlen=24 * 60)
    history_counter = 0

    history_lock = threading.Lock()

    def history_thread():
        next_timestamp = datetime.now()
        while True:
            next_timestamp += timedelta(minutes=60)
            with history_lock:
                nonlocal history_counter
                history_counter = history_counter + 1
                history.append(binance_account.mark_prices.copy())
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
