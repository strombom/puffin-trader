
import csv
import pickle
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression


def string_to_timestamp(date):
    return datetime.timestamp(datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f"))

class Event:
    def __init__(self, timestamp, price):
        self.timestamp = timestamp
        self.price = price

    def __repr__(self):
        return f"Event({self.timestamp}, {self.price})"

def read_events(settings):
    try:
        with open('cache_events.pickle', 'rb') as f:
            data = pickle.load(f)
            if data['start_timestamp'] == settings['start_timestamp'] and data['end_timestamp'] == settings['end_timestamp']:
                print(datetime.fromtimestamp(data['data'][0].timestamp), '-', datetime.fromtimestamp(data['data'][-1].timestamp))
                return data['data']
    except:
        pass

    events = []
    with open(settings['events_filepath']) as csv_file:
        for row in csv.reader(csv_file):
            ts = (settings['data_first_timestamp'] * 1000 + int(row[0])) / 1000
            if ts > settings['start_timestamp']:
                events.append(Event(ts, float(row[1])))
            if ts > settings['end_timestamp']:
                break

    with open('cache_events.pickle', 'wb') as f:
        data = {
            'start_timestamp': settings['start_timestamp'],
            'end_timestamp': settings['end_timestamp'],
            'data': events
        }
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    print(datetime.fromtimestamp(events[0].timestamp), '-', datetime.fromtimestamp(events[-1].timestamp))

    return events

def calc_volatilities(events, settings):
    try:
        with open('cache_volatilities.pickle', 'rb') as f:
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

    with open('cache_volatilities.pickle', 'wb') as f:
        data = {
            'start_timestamp': settings['start_timestamp'],
            'end_timestamp': settings['end_timestamp'],
            'data': volatilities
        }
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return volatilities


def calc_volatilities_regr(events, settings):
    try:
        with open('cache_volatilities_regr.pickle', 'rb') as f:
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

    with open('cache_volatilities_regr.pickle', 'wb') as f:
        data = {
            'start_timestamp': settings['start_timestamp'],
            'end_timestamp': settings['end_timestamp'],
            'data': volatilities
        }
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return volatilities

def calc_directions(events, settings):
    try:
        with open('cache_directions.pickle', 'rb') as f:
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

    with open('cache_directions.pickle', 'wb') as f:
        data = {
            'start_timestamp': settings['start_timestamp'],
            'end_timestamp': settings['end_timestamp'],
            'data': (directions, velocities)
        }
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return directions, velocities
