import json

from binance.client import Client

from binance_account import BinanceAccount


top_symbols = [
    'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOT', 'LTC', 'XLM', 'BCH', 'THETA', 'FIL', 'TRX', 'DOGE', 'VET', 'SOL', 'EOS',
    'XMR', 'LUNA', 'IOTA', 'XTZ', 'ATOM', 'NEO', 'AVAX', 'ALGO', 'EGLD', 'XEM', 'KSM', 'DASH', 'HBAR', 'ZEC', 'DCR',
    'STX', 'NEAR', 'ETC', 'ZIL', 'BTG', 'TFUEL', 'WAVES', 'ICX', 'RVN', 'ONT', 'QTUM', 'ONE', 'HNT', 'SC', 'DGB',
    'FTM', 'IOST', 'CELO', 'ZEN', 'LSK', 'NANO', 'CKB', 'XVG', 'NKN', 'BCD', 'ARDR', 'IOTX', 'STEEM', 'KMD', 'WAN',
    'STRAX', 'SRM', 'BTS', 'ARK', 'IRIS', 'ROSE', 'SYS', 'SCRT', 'HIVE', 'TOMO', 'BTM', 'ELA', 'AION', 'CTC', 'CCXX',
    'DFI', 'XWC', 'XDC', 'BCHA', 'ETN', 'AKT', 'HTR', 'ARRR', 'CRU', 'MARO', 'EDG', 'COTI', 'HNC', 'RDD', 'META',
    'LTO', 'VRA', 'BCN', 'MWC', 'PHA', 'REV', 'KDA', 'NRG', 'APL', 'MONA', 'PAC', 'TT', 'NYE', 'FIRO', 'SAPP'
]


if __name__ == "__main__":
    with open('binance_account.json') as f:
        account_info = json.load(f)
        api_key = account_info['api_key']
        api_secret = account_info['api_secret']

    binance_client = Client(api_key, api_secret)
    binance_account = BinanceAccount(binance_client=binance_client, symbols=top_symbols)
