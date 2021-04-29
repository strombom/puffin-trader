import zmq
import json
from time import sleep
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


if __name__ == "__main__":
    with open('binance_account.json') as f:
        account_info = json.load(f)
        api_key = account_info['api_key']
        api_secret = account_info['api_secret']

    binance_client = Client(api_key, api_secret)
    binance_account = BinanceAccount(binance_client=binance_client, top_symbols=top_symbols)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:31007")

    while True:
        message = socket.recv_pyobj()
        print("Received request: %s" % message)
        socket.send_pyobj(binance_account.mark_prices)
