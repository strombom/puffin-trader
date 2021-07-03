import csv
import pickle
from datetime import datetime
from datetime import timezone


class AggTick:
    def __init__(self, timestamp, low, high):
        self.timestamp = timestamp
        self.low = low
        self.high = high
        self.mid = (low + high) / 2

    def __repr__(self):
        return f'AggTick({self.timestamp} {self.low}, {self.high})'


def read_agg_ticks(path, filenames, start_timestamp=None, end_timestamp=None, ignore_date_ranges=[]):
    try:
        with open(f"cache/agg_ticks.pickle", 'rb') as f:
            data = pickle.load(f)
            return data
    except:
        pass

    #timestamp_start = string_to_datetime("2020-01-01 00:00:00.000")
    #timestamp_end = string_to_datetime("2020-11-01 00:00:00.000")

    agg_ticks = []
    prev_ask, prev_bid = 0, 0
    for filename in filenames:
        with open(path + filename, 'r') as csv_file:
            for row in csv.reader(csv_file):
                timestamp = datetime.fromtimestamp(float(row[0]) / 1000, tz=timezone.utc)
                if start_timestamp is not None and timestamp < start_timestamp:
                    continue
                if end_timestamp is not None and timestamp >= end_timestamp:
                    break

                ignore = False
                for start_ignore, end_ignore in ignore_date_ranges:
                    if start_ignore < timestamp < end_ignore:
                        ignore = True
                        break
                if ignore:
                    continue

                ask, bid = float(row[1]), float(row[2])
                if ask != prev_ask or bid != prev_bid:
                    agg_ticks.append(AggTick(timestamp=timestamp, low=bid, high=ask))
                    prev_ask, prev_bid = ask, bid
                #if len(agg_ticks) > 1000000:
                #    break

    with open(f"cache/agg_ticks.pickle", 'wb') as f:
        data = agg_ticks
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return agg_ticks
