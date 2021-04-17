import os
import glob
import pandas as pd
from datetime import datetime

from IntrinsicTime.runner import Runner


if __name__ == '__main__':
    start_date = datetime.strptime('2021-01-01 00:00:00 UTC', '%Y-%m-%d %H:%M:%S %Z')
    end_date = datetime.strptime('2021-04-14 00:00:00 UTC', '%Y-%m-%d %H:%M:%S %Z')
    delta = 0.005

    pairs = []
    for file_path in glob.glob("cache/tickers/*.csv"):
        pair = os.path.basename(file_path).replace('.csv', '')
        pairs.append(pair)

    for pair in pairs:
        data = pd.read_csv(f"cache/tickers/{pair}.csv")
        pair_start_date = datetime.utcfromtimestamp(data.iloc[0]['timestamp'] / 1000)
        pair_end_date = datetime.utcfromtimestamp(data.iloc[data.shape[0] - 1]['timestamp'] / 1000)
        if pair_start_date != start_date or pair_end_date < end_date:
            continue
        dirs = []
        runner = Runner(delta=delta)

        timestamps, ie_events = [], []
        for idx, row in data.iterrows():
            for ie_event in runner.step(high=row['high'], low=row['low']):
                timestamps.append(row['timestamp'])
                ie_events.append(ie_event)
                
        print(ie_events)
        print(len(ie_events))
        print(data.shape)
        print(pair)
        quit()
