
import os, csv
import pickle
from datetime import datetime, timedelta, timezone

from Common.Misc import string_to_datetime


def make_data():
    file_path = f"C:/Users/jst/Downloads/bitstampUSD.csv/.bitstampUSD.csv"
    last_hour = string_to_datetime("2011-09-14 00:00:00.0")

    timestamps, prices = [], []
    with open(file_path, 'r') as csv_file:
        for row in csv.reader(csv_file):
            timestamp = datetime.fromtimestamp(float(row[0]), tz=timezone.utc)

            if timestamp > last_hour:
                price = float(row[1])
                prices.append(price)
                timestamps.append(timestamp)
                last_hour += timedelta(hours=1)

    return timestamps, prices


def get_data():
    with open(f"cache/bitstamp_hourly.pickle", 'rb') as f:
        timestamps, prices = pickle.load(f)

    return timestamps, prices


if __name__ == '__main__':
    timestamps, prices = make_data()

    if not os.path.exists('cache'):
        os.makedirs('cache')

    with open(f"cache/bitstamp_hourly.pickle", 'wb') as f:
        pickle.dump((timestamps, prices), f, pickle.HIGHEST_PROTOCOL)
