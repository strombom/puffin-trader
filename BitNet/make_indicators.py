import queue
import pickle
import pathlib
import warnings
import numpy as np
import pandas as pd
from numba import jit
import multiprocessing
from datetime import datetime, timezone, timedelta

from cache import cache_it

warnings.simplefilter('ignore', np.RankWarning)


@jit(nopython=False)
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

    return directions


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
        start_timestamp = parameters['start_timestamp']
        direction_degrees = parameters['direction_degrees']
        n_timesteps = parameters['n_timesteps']

        print(f"Processing {symbol}")

        indicators = np.zeros((len(lengths), len(direction_degrees), n_timesteps))

        ie_timestamps = np.array(intrinsic_events[symbol]['timestamps'])
        ie_prices = np.array(intrinsic_events[symbol]['prices'])

        for direction_degree in direction_degrees:
            directions = np.zeros((len(lengths), ie_timestamps.shape[0]))

            directions = make_spectrum(
                lengths=lengths,
                prices=ie_prices,
                poly_order=direction_degree,
                directions=directions
            )

            ie_idx = 0
            first_timestamp = int((ie_timestamps[ie_idx] - start_timestamp).total_seconds() / 60)
            next_timestamp = int((ie_timestamps[ie_idx + 1] - start_timestamp).total_seconds() / 60)

            for indicator_idx in range(first_timestamp, n_timesteps):
                while indicator_idx >= next_timestamp and ie_idx + 1 < directions.shape[1]:
                    ie_idx += 1
                    next_timestamp = int((ie_timestamps[ie_idx + 1] - start_timestamp).total_seconds() / 60)
                indicators[:, direction_degree - 1, indicator_idx] = directions[:, ie_idx]

        path = f"cache/indicators"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        with open(f"{path}/{symbol}.pickle", 'wb') as f:
            pickle.dump({
                'lengths': lengths,
                #'directions': directions,
                'indicators': indicators
            }, f)


def make_indicators(start_timestamp: datetime, end_timestamp: datetime):
    with open(f"cache/intrinsic_events.pickle", 'rb') as f:
        intrinsic_events = pickle.load(f)

    direction_degrees = [1, 2, 3]
    lengths = np.array([5, 7, 11, 15, 22, 33, 47, 68, 100, 150])
    lengths_df = pd.DataFrame(data={'length': lengths})
    lengths_df.to_csv('cache/regime_data_lengths.csv')

    n_timesteps = int((end_timestamp - start_timestamp).total_seconds() / 60)

    task_queue = multiprocessing.Queue()
    lock = multiprocessing.Lock()

    """
    for symbol in intrinsic_events:
        task_queue.put({
            'symbol': symbol,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'lengths': list(lengths),
            'direction_degrees': direction_degrees,
            'n_timesteps': n_timesteps
        })
        make_indicator(lock, task_queue)
        break

    quit()
    """

    for symbol in intrinsic_events:
        task_queue.put({
            'symbol': symbol,
            'start_timestamp': start_timestamp,
            'lengths': list(lengths),
            'direction_degrees': direction_degrees,
            'n_timesteps': n_timesteps
        })

    ps = []
    for n in range(min(multiprocessing.cpu_count() - 5, len(intrinsic_events))):
        print("New thread", n)
        p = multiprocessing.Process(target=make_indicator, args=(lock, task_queue))
        ps.append(p)
        p.start()

    for p in ps:
        p.join()


if __name__ == '__main__':
    start_timestamp_ = datetime.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_timestamp_ = datetime.strptime("2021-07-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    make_indicators(start_timestamp=start_timestamp_, end_timestamp=end_timestamp_)
