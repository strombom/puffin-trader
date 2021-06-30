import queue
import pickle
import pathlib
import warnings
import numpy as np
import pandas as pd
from numba import jit
import multiprocessing
from datetime import datetime, timezone, timedelta

warnings.simplefilter('ignore', np.RankWarning)


#@jit(nopython=False)
def make_spectrum(lengths, prices, poly_order, directions):
    for length_idx, length in enumerate(lengths):
        for idx in range(lengths[-1], prices.shape[0]):
            # TODO Is the last price not used?
            start, end = idx - length, idx
            xp = np.arange(start, end)
            yp = np.poly1d(np.polyfit(xp, prices[start:end], poly_order))  # TODO end+1 ?

            curve = yp(xp)
            direction = curve[-1] / curve[-2] - 1.0
            directions[length_idx, idx] = direction


def make_indicator(lock, task_queue):
    with open(f"cache/intrinsic_events.pickle", 'rb') as f:
        intrinsic_events = pickle.load(f)

    while True:
        try:
            with lock:
                parameters = task_queue.get(block=True, timeout=1)
        except queue.Empty:
            break

        symbol = parameters['symbol']
        lengths = parameters['lengths']
        #start_timestamp = parameters['start_timestamp']
        #end_timestamp = parameters['end_timestamp']
        direction_degrees = parameters['direction_degrees']
        n_timesteps = parameters['n_timesteps']

        indicators = np.zeros((len(lengths), len(direction_degrees), n_timesteps))

        steps_idx = np.array(intrinsic_events[symbol]['klines_idxs'])
        steps_price = np.array(intrinsic_events[symbol]['steps_price'])

        print(f"Processing {symbol}")

        for direction_degree in direction_degrees:
            directions = np.zeros((len(lengths), steps_idx.shape[0]))

            make_spectrum(lengths=lengths,
                          prices=steps_price,
                          poly_order=direction_degree,
                          directions=directions)

            direction_idx = 0
            for indicator_idx in range(steps_idx[0], n_timesteps):
                while indicator_idx > steps_idx[direction_idx] and direction_idx + 1 < len(steps_idx):
                    direction_idx += 1
                indicators[:, direction_degree - 1, indicator_idx] = directions[:, direction_idx]

        path = f"cache/indicators"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        with open(f"{path}/{symbol}.pickle", 'wb') as f:
            pickle.dump({
                'lengths': lengths,
                #'directions': directions,
                'indicators': indicators
            }, f)


def main():
    with open(f"cache/intrinsic_events.pickle", 'rb') as f:
        intrinsic_events = pickle.load(f)

    start_timestamp = datetime.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_timestamp = datetime.strptime("2021-06-29 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    direction_degrees = [1, 2, 3]
    lengths = np.array([5, 7, 11, 15, 22, 33, 47, 68, 100])
    lengths_df = pd.DataFrame(data={'length': lengths})
    lengths_df.to_csv('cache/regime_data_lengths.csv')

    n_timesteps = int((end_timestamp - start_timestamp).total_seconds() / 60)

    task_queue = multiprocessing.Queue()
    lock = multiprocessing.Lock()

    for symbol in intrinsic_events:
        task_queue.put({
            'symbol': symbol,
            #"'start_timestamp': start_timestamp,
            #'end_timestamp': end_timestamp,
            'lengths': list(lengths),
            'direction_degrees': direction_degrees,
            'n_timesteps': n_timesteps
        })

    ps = []
    for n in range(min(multiprocessing.cpu_count() - 3, len(intrinsic_events))):
        print("New thread", n)
        p = multiprocessing.Process(target=make_indicator, args=(lock, task_queue))
        ps.append(p)
        p.start()

    for p in ps:
        p.join()


if __name__ == '__main__':
    main()
