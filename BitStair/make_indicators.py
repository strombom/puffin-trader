import os
import glob
import pandas as pd
from datetime import datetime

from IntrinsicTime.runner import Runner


if __name__ == '__main__':

    pairs = []
    for file_path in glob.glob("cache/tickers/*.csv"):
        pair = os.path.basename(file_path).replace('.csv', '')
        pairs.append(pair)

    start_date = datetime.strptime('2021-01-01 00:00:00 UTC', '%Y-%m-%d %H:%M:%S %Z')

    for pair in pairs:
        data = pd.read_csv(f"cache/tickers/{pair}.csv")
        pair_start_date = datetime.utcfromtimestamp(data.iloc[0]['timestamp'] / 1000)
        if pair_start_date == start_date:
            #print("Ok")
            pass
        else:
            print("nope", pair)

    delta = 0.001

    #    runner = Runner(delta=delta, order_book=order_books[0])
    #    quit()

