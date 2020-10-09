
import csv
import pickle
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression


def string_to_timestamp(date):
    return datetime.timestamp(datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f'))


def timestamp_to_string(timestamp):
    return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')


class Coastline:
    def __init__(self, directions, timestamps_delta, timestamps_overshoot, prices_delta, prices_overshoot):
        self.directions = directions
        self.timestamps_delta = timestamps_delta
        self.timestamps_overshoot = timestamps_overshoot
        self.prices_delta = prices_delta
        self.prices_overshoot = prices_overshoot

    def __repr__(self):
        return f"CoastLine({timestamp_to_string(self.timestamps_delta[0])} {self.directions.shape[0]})"


class AggTick:
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.mid = (low + high) / 2

    def __repr__(self):
        return f'AggTick({self.low}, {self.high})'


def read_agg_ticks(filepath):
    try:
        with open(f"cache/agg_ticks.pickle", 'rb') as f:
            data = pickle.load(f)
            return data
    except:
        pass

    agg_ticks = []
    with open(filepath, 'r') as csv_file:
        for row in csv.reader(csv_file):
            agg_ticks.append(AggTick(low=float(row[2]), high=float(row[1])))

    with open(f"cache/agg_ticks.pickle", 'wb') as f:
        data = agg_ticks
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return agg_ticks


def read_coastlines(settings):
    coastlines = {}
    try:
        with open(f"cache/coastlines.pickle", 'rb') as f:
            data = pickle.load(f)
            if data['start_timestamp'] == settings['start_timestamp'] and \
               data['end_timestamp'] == settings['end_timestamp'] and \
               data['deltas'] == settings['deltas']:
                #print(datetime.fromtimestamp(data['data'][0].timestamp), '-', datetime.fromtimestamp(data['data'][-1].timestamp))
                return data['data']
    except:
        pass

    for delta in settings['deltas']:
        # PD_Direction direction,
        # time_point_ms timestamp_delta, time_point_ms timestamp_overshoot,
        # float price_delta, float price_overshoot,
        # size_t agg_tick_idx_delta, size_t agg_tick_idx_overshoot

        directions = []
        timestamps_delta = []
        timestamps_overshoot = []
        prices_delta = []
        prices_overshoot = []
        with open(settings['events_filepath'] + f"_{delta:.6f}.csv") as csv_file:
            for row in csv.reader(csv_file):
                direction = int(row[0])
                ts_delta = (settings['data_first_timestamp'] * 1000 + int(row[1])) / 1000
                ts_overshoot = (settings['data_first_timestamp'] * 1000 + int(row[2])) / 1000
                price_delta = float(row[3])
                price_overshoot = float(row[4])

                if ts_delta > settings['start_timestamp'] and ts_overshoot < settings['end_timestamp']:
                    directions.append(direction)
                    timestamps_delta.append(ts_delta)
                    prices_delta.append(price_delta)
                    timestamps_overshoot.append(ts_overshoot)
                    prices_overshoot.append(price_overshoot)
                if ts_overshoot > settings['end_timestamp']:
                    break

        directions = np.array(directions)
        timestamps_delta = np.array(timestamps_delta)
        timestamps_overshoot = np.array(timestamps_overshoot)
        prices_delta = np.array(prices_delta)
        prices_overshoot = np.array(prices_overshoot)

        coastlines[delta] = Coastline(directions, timestamps_delta, timestamps_overshoot, prices_delta, prices_overshoot)

        with open(f"cache/coastlines.pickle", 'wb') as f:
            data = {
                'start_timestamp': settings['start_timestamp'],
                'end_timestamp': settings['end_timestamp'],
                'deltas': settings['deltas'],
                'data': coastlines
            }
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return coastlines


def calc_volatilities(events, settings):
    try:
        with open(f"cache/volatilities_{settings['volatility_buffer_length']}.pickle", 'rb') as f:
            data = pickle.load(f)
            if data['start_timestamp'] == settings['start_timestamp'] and data['end_timestamp'] == settings['end_timestamp']:
                return data['data']
    except:
        pass

    buffer_length = settings['volatility_buffer_length']
    vola_prices = np.ones((buffer_length, 1)) * events[0].price
    volatilities = []

    for event in events:
        vola_prices[:-1] = vola_prices[1:]
        vola_prices[-1] = event.price
        volatility = max(vola_prices) / min(vola_prices) - 1
        volatilities.append(volatility)
    volatilities = np.array(volatilities)[:, 0]

    with open(f"cache/volatilities_{settings['volatility_buffer_length']}.pickle", 'wb') as f:
        data = {
            'start_timestamp': settings['start_timestamp'],
            'end_timestamp': settings['end_timestamp'],
            'data': volatilities
        }
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return volatilities


def calc_volatilities_regr(events, settings):
    try:
        with open(f"cache/volatilities_regr_{settings['volatility_buffer_length']}.pickle", 'rb') as f:
            data = pickle.load(f)
            if data['start_timestamp'] == settings['start_timestamp'] and data['end_timestamp'] == settings['end_timestamp']:
                return data['data']
    except:
        pass

    buffer_length = settings['volatility_buffer_length']
    x = np.arange(buffer_length).reshape((buffer_length, 1))
    vola_prices = np.ones((buffer_length, 1)) * events[0].price
    volatilities = []

    for event in events:
        vola_prices[:-1] = vola_prices[1:]
        vola_prices[-1] = event.price
        regressor = LinearRegression()
        regressor.fit(x, vola_prices)  # actually produces the linear eqn for the data
        y = regressor.intercept_ + regressor.coef_ * x
        z = vola_prices - y + np.mean(vola_prices)
        volatility = max(z) / min(z) - 1
        volatilities.append(volatility)
    volatilities = np.array(volatilities)[:, 0]

    with open(f"cache/volatilities_regr_{settings['volatility_buffer_length']}.pickle", 'wb') as f:
        data = {
            'start_timestamp': settings['start_timestamp'],
            'end_timestamp': settings['end_timestamp'],
            'data': volatilities
        }
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return volatilities

def calc_directions(events, settings):
    try:
        with open(f"cache/directions_{settings['volatility_buffer_length']}.pickle", 'rb') as f:
            data = pickle.load(f)
            if data['start_timestamp'] == settings['start_timestamp'] and data['end_timestamp'] == settings['end_timestamp']:
                return data['data']
    except:
        pass

    buffer_length = settings['volatility_buffer_length']
    x = np.arange(buffer_length).reshape((buffer_length, 1))
    dir_prices = np.ones((buffer_length, 1)) * events[0].price
    directions = []

    for event in events:
        dir_prices[:-1] = dir_prices[1:]
        dir_prices[-1] = event.price
        regressor = LinearRegression()
        regressor.fit(x, dir_prices)  # actually produces the linear eqn for the data
        direction = regressor.coef_[0]
        directions.append(direction)
    directions = np.array(directions)[:, 0]

    velocities = np.zeros(directions.shape[0])
    velocities[2:] = directions[2:] - directions[:-2]

    with open(f"cache/directions_{settings['volatility_buffer_length']}.pickle", 'wb') as f:
        data = {
            'start_timestamp': settings['start_timestamp'],
            'end_timestamp': settings['end_timestamp'],
            'data': (directions, velocities)
        }
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return directions, velocities
