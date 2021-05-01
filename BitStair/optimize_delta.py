import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime

from IntrinsicTime.runner import Runner


if __name__ == '__main__':
    plot = False

    start_date = datetime.strptime('2021-03-01 00:00:00 UTC', '%Y-%m-%d %H:%M:%S %Z')
    end_date = datetime.strptime('2021-03-07 00:00:00 UTC', '%Y-%m-%d %H:%M:%S %Z')

    pairs = []
    for file_path in glob.glob("cache/tickers/*.csv"):
        pair = os.path.basename(file_path).replace('.csv', '')
        pairs.append(pair)

    deltas = np.arange(start=0.005, stop=0.025, step=0.003)
    steps = np.zeros((len(pairs), deltas.shape[0]))

    for pair_idx, pair in enumerate(pairs):
        data = pd.read_csv(f"cache/tickers/{pair}.csv")

        for delta_idx, delta in enumerate(deltas):
            runner = Runner(delta=delta)

            runner_length = 0
            for idx, row in data.iterrows():
                ie_events = runner.step(high=row['high'], low=row['low'])
                runner_length += len(ie_events)

            steps[pair_idx, delta_idx] = runner_length
            print(pair, delta, runner_length)

    steps = pd.DataFrame(steps)
    steps.to_csv(path_or_buf="cache/steps.csv")
