import os
import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from fastai.learner import load_learner


class ProfitModel:
    def __init__(self):
        self.model_idx = 0
        self.model_files = []
        for filename in glob.glob('E:/BitBot/models/model_*_*.pickle'):
            timestamp = datetime.strptime(filename[-17:-7], "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(hours=6)
            self.model_files.append({'timestamp': timestamp, 'filename': filename})
        self.model_files = sorted(self.model_files, key=lambda x: x['timestamp'])

        self.model = load_learner(self.model_files[self.model_idx]['filename'])

    def has_new_model(self, timestamp):
        if self.model_idx + 1 < len(self.model_files) and timestamp >= self.model_files[self.model_idx + 1]['timestamp']:
            return True
        else:
            return False

    def predict(self, indicators):
        test_dl = self.model.dls.test_dl(indicators)
        return self.model.get_preds(dl=test_dl)[0][:, 0].numpy()

    def load_next_model(self):
        self.model = load_learner(self.model_files[self.model_idx]['filename'])
        self.model_idx += 1

    def last_predictable_timestamp(self):
        timespan = self.model_files[1]['timestamp'] - self.model_files[0]['timestamp']
        return self.model_files[-1]['timestamp'] + timespan


class Indicators:
    def __init__(self, path, symbols):
        self.indicators = {}
        self.idx = {}
        self.next_timestamp = {}

        for symbol in symbols:
            file_path = os.path.join(path, f"{symbol}.csv")
            with open(file_path, 'rb') as f:
                self.indicators[symbol] = pd.read_csv(f, parse_dates=[0])
            self.idx[symbol] = 0
            self.next_timestamp[symbol] = self.indicators[symbol].iloc[0]['timestamp']

    def get_start_end_date(self):
        start_date, end_date = None, None
        for symbol in self.indicators:
            first_date = self.indicators[symbol].iloc[0]['timestamp'].to_pydatetime()
            last_date = self.indicators[symbol].iloc[-1]['timestamp'].to_pydatetime()
            if start_date is None:
                start_date = first_date
            if end_date is None:
                end_date = last_date
            start_date = max(start_date, first_date)
            end_date = max(end_date, last_date)
        return start_date, end_date

    def get_next_section(self, symbol, timestamp):
        start_idx = self.idx[symbol]
        end_idx = start_idx
        while end_idx + 1 < self.indicators[symbol].shape[0]:
            if self.indicators[symbol]['timestamp'].iloc[end_idx + 1] >= timestamp:
                break
            end_idx += 1
        self.idx[symbol] = end_idx
        return self.indicators[symbol].iloc[start_idx:end_idx]


def make_predictions():
    training_path = 'E:/BitBot/training_data/'
    symbols = set()
    for filename in os.listdir(training_path):
        symbols.add(filename.replace('.csv', ''))
    symbols = list(symbols)

    profit_model = ProfitModel()
    indicators = Indicators(training_path, symbols)
    raw_predictions = {symbol: [] for symbol in symbols}

    timestamp_start, timestamp_end = indicators.get_start_end_date()
    timestamp_end = min(timestamp_end, profit_model.last_predictable_timestamp())
    timestamp_end = datetime(year=2021, month=1, day=10, tzinfo=timezone.utc)

    timestamp = timestamp_start
    while timestamp < timestamp_end:
        if profit_model.has_new_model(timestamp):
            for symbol in symbols:
                section = indicators.get_next_section(symbol, timestamp)
                raw_predictions[symbol].append(profit_model.predict(section))
                print(f"Predict {symbol} {timestamp}")
            profit_model.load_next_model()
        timestamp += timedelta(minutes=1)

    for symbol in symbols:
        section = indicators.get_next_section(symbol, timestamp_end)
        raw_predictions[symbol].append(profit_model.predict(section))
        print(f"Predict {symbol} {timestamp_end}")

    for symbol in symbols:
        raw_predictions[symbol] = np.concatenate(raw_predictions[symbol])

    predictions = []
    prediction_idx = {symbol: 0 for symbol in symbols}
    timestamp = timestamp_start
    timestamp_print = timestamp + timedelta(days=1)
    while timestamp < timestamp_end:
        prediction = {'timestamp': timestamp}
        for symbol in symbols:
            while prediction_idx[symbol] < raw_predictions[symbol].shape[0] and timestamp >= indicators.indicators[symbol].iloc[prediction_idx[symbol]]['timestamp']:
                prediction[symbol] = raw_predictions[symbol][prediction_idx[symbol]]
                prediction_idx[symbol] += 1
        predictions.append(prediction)
        timestamp += timedelta(minutes=1)

        if timestamp >= timestamp_print:
            print(f"Compiling predictions {timestamp}")
            timestamp_print = timestamp + timedelta(days=1)

    file_path = f"cache/predictions.pickle"
    with open(file_path, 'wb') as f:
        pickle.dump({
            'symbols': symbols,
            'predictions': predictions
        }, f)


if __name__ == '__main__':
    make_predictions()
