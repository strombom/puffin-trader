import glob
import os
import json

import pandas as pd
from binance.client import Client
from pandas import DataFrame


top_symbols = [
    'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOT', 'LTC', 'XLM', 'BCH', 'THETA', 'FIL', 'TRX', 'DOGE', 'VET', 'SOL', 'EOS',
    'XMR', 'LUNA', 'IOTA', 'XTZ', 'ATOM', 'NEO', 'AVAX', 'ALGO', 'EGLD', 'XEM', 'KSM', 'DASH', 'HBAR', 'ZEC', 'DCR',
    'STX', 'NEAR', 'ETC', 'ZIL', 'BTG', 'TFUEL', 'WAVES', 'ICX', 'RVN', 'ONT', 'QTUM', 'ONE', 'HNT', 'SC', 'DGB',
    'FTM', 'IOST', 'CELO', 'ZEN', 'LSK', 'NANO', 'CKB', 'XVG', 'NKN', 'BCD', 'ARDR', 'IOTX', 'STEEM', 'KMD', 'WAN',
    'STRAX', 'SRM', 'BTS', 'ARK', 'IRIS', 'ROSE', 'SYS', 'SCRT', 'HIVE', 'TOMO', 'BTM', 'ELA', 'AION', 'CTC', 'CCXX',
    'DFI', 'XWC', 'XDC', 'BCHA', 'ETN', 'AKT', 'HTR', 'ARRR', 'CRU', 'MARO', 'EDG', 'COTI', 'HNC', 'RDD', 'META',
    'LTO', 'VRA', 'BCN', 'MWC', 'PHA', 'REV', 'KDA', 'NRG', 'APL', 'MONA', 'PAC', 'TT', 'NYE', 'FIRO', 'SAPP'
]


class BinanceAccount:
    assets = {}
    mark_price_ask = 0
    mark_price_bid = 0

    def __init__(self, client: Client):
        self.client = client

    def get_tickers(self):
        file_path = 'cache/tickers.csv'
        if os.path.exists(file_path):
            pairs = pd.read_csv(file_path)
            return pairs

        all_tickers = self.client.get_all_tickers()
        base_symbol = 'USDT'
        pairs = []
        for ticker in all_tickers:
            pair = ticker['symbol']
            if not pair.endswith(base_symbol):
                continue
            if pair.replace(base_symbol, '') not in top_symbols:
                continue
            pairs.append({'pair': pair, 'symbol': pair.replace(base_symbol, ''), 'base': base_symbol})
        pairs = DataFrame(pairs)
        pairs.to_csv(file_path)
        return pairs


def download_klines(pair):
    klines = []
    for kline in binance_client.get_historical_klines_generator(pair, Client.KLINE_INTERVAL_1MINUTE, "2021-04-18 UTC"):
        klines.append({
            'timestamp': kline[0],
            'close': kline[4],
            'high': kline[2],
            'low': kline[3],
            'volume': kline[5]
        })
    klines = DataFrame(klines)
    file_path = f"cache/tickers/{pair}.csv"
    klines.to_csv(file_path)
    print(f"Saved {file_path}")


def crop_tickers():
    # Make sure all tickers have exactly the same length

    pairs = []
    for file_path in glob.glob("cache/tickers/*.csv"):
        pair = os.path.basename(file_path).replace('.csv', '')
        pairs.append(pair)

    min_length = None
    for pair in pairs:
        data = pd.read_csv(f"cache/tickers/{pair}.csv")

        if min_length is None:
            min_length = data.shape[0]
        min_length = min(min_length, data.shape[0])

    for pair in pairs:
        data = pd.read_csv(f"cache/tickers/{pair}.csv")
        data = data.iloc[0:min_length, :]
        data.to_csv(f"cache/tickers/{pair}.csv")


if __name__ == '__main__':
    with open('binance_account.json') as f:
        account_info = json.load(f)
        api_key = account_info['api_key']
        api_secret = account_info['api_secret']

    binance_client = Client(api_key, api_secret)
    binance_account = BinanceAccount(client=binance_client)
    tickers = binance_account.get_tickers()

    for ticker_idx, ticker in tickers.iterrows():
        download_klines(pair=ticker['pair'])

    crop_tickers()
