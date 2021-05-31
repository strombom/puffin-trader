import os
import json
import pandas as pd
import pyrate_limiter
from datetime import datetime
from binance.client import Client


def download_klines(client: Client, symbol: str, start_time: str):
    file_path = f"cache/klines/{symbol}.hdf"
    print(f"Downloading {file_path}")

    if os.path.exists(file_path):
        return

    limiter = pyrate_limiter.Limiter(pyrate_limiter.RequestRate(1, pyrate_limiter.Duration.SECOND))

    @limiter.ratelimit("download_print_status", delay=False)
    def print_status(timestamp):
        print(f"{datetime.now()} {timestamp}")

    klines = []
    for kline in client.get_historical_klines_generator(symbol, Client.KLINE_INTERVAL_1MINUTE, start_time):
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
            print_status(timestamp=datetime.fromtimestamp(klines[-1]['close_time'] / 1000))
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

    symbol = "BTCUSDT"
    download_klines(client=client, symbol=symbol, start_time=start_time)

    quit()

    for symbol in exchange_info['symbols']:
        if symbol['symbol'].endswith('USDT') and symbol['isMarginTradingAllowed'] and 'MARGIN' in symbol['permissions']:
            download_klines(client=client, symbol=symbol['symbol'], start_time=start_time)


if __name__ == '__main__':
    main()
