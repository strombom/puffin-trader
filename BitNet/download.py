import os
import json
import pathlib

import pandas as pd
import pyrate_limiter
from datetime import datetime, timezone
from binance.client import Client


def download_klines(client: Client, symbol: str, start_time: str):
    path = f"cache/klines_new"
    file_path = f"{path}/{symbol}.hdf"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    print(f"Downloading {file_path}")

    last_open_time = 0
    klines = []
    if os.path.exists(file_path):
        old_klines = pd.read_hdf(path_or_buf=file_path)
        last_open_time = old_klines.iloc[-1]['open_time']
        start_time = datetime.fromtimestamp(last_open_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d UTC")
        klines = old_klines.to_dict('records')

    limiter = pyrate_limiter.Limiter(pyrate_limiter.RequestRate(1, pyrate_limiter.Duration.SECOND))

    @limiter.ratelimit("download_print_status", delay=False)
    def print_status(timestamp):
        print(f"{datetime.now()} {timestamp}")

    for kline in client.get_historical_klines_generator(symbol, Client.KLINE_INTERVAL_1MINUTE, start_time):
        if kline[0] <= last_open_time:
            continue
        klines.append({
            'open_time': kline[0],
            'open': float(kline[1]),
            'high': float(kline[2]),
            'low': float(kline[3]),
            'close': float(kline[4]),
            'volume': float(kline[5]),
            'close_time': kline[6],
            'number_of_trades': kline[8]
        })

        try:
            print_status(timestamp=datetime.fromtimestamp(klines[-1]['close_time'] / 1000, tz=timezone.utc))
        except pyrate_limiter.BucketFullException as err:
            pass

    klines = pd.DataFrame(klines)
    klines.to_hdf(
        path_or_buf=file_path,
        key=symbol,
        mode='w',
        complevel=9,
        complib='blosc'
    )


def main():
    with open('binance_account.json') as f:
        account_info = json.load(f)

    start_time = "2020-01-01 UTC"

    client = Client(
        api_key=account_info['api_key'],
        api_secret=account_info['api_secret']
    )

    exchange_info = client.get_exchange_info()

    #symbol = "BTCUSDT"
    #download_klines(client=client, symbol=symbol, start_time=start_time)

    #quit()

    #symbols = {}

    for base_symbol in ['USDT', 'BUSD', 'USDC']:
        #symbols[base_symbol] = []
        for symbol in exchange_info['symbols']:
            if symbol['symbol'].endswith(base_symbol) and symbol['isMarginTradingAllowed'] and 'MARGIN' in symbol['permissions']:
                #symbols[base_symbol].append(symbol['symbol'])
                download_klines(client=client, symbol=symbol['symbol'], start_time=start_time)

    #for base_symbol in symbols:
    #    print(len(symbols[base_symbol]), symbols[base_symbol])


if __name__ == '__main__':
    main()
