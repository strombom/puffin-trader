import os
import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from fastai.learner import load_learner


class ProfitModel:
    def __init__(self):
        training_time = timedelta(minutes=10)
        self.model_files = []
        for filename in glob.glob('E:/BitBot/models/model_*_*.pickle'):
            timestamp = datetime.strptime(filename[-17:-7], "%Y-%m-%d").replace(tzinfo=timezone.utc) + training_time
            self.model_files.append({'timestamp': timestamp, 'filename': filename})
        self.model_files = sorted(self.model_files, key=lambda x: x['timestamp'])

        self.model_idx = -1
        self.load_next_model()
        #self.model = load_learner(self.model_files[self.model_idx]['filename'])

    def has_new_model(self, timestamp):
        if self.model_idx + 1 < len(self.model_files) and timestamp >= self.model_files[self.model_idx + 1]['timestamp']:
            return True
        else:
            return False

    def predict(self, indicators):
        test_dl = self.model.dls.test_dl(indicators)
        pred = self.model.get_preds(dl=test_dl)[0].numpy()
        return pred

    def load_next_model(self):
        self.model_idx += 1
        #print("Loading learner", self.model_files[self.model_idx]['filename'])
        self.model = load_learner(self.model_files[self.model_idx]['filename'])

    def last_predictable_timestamp(self):
        timespan = self.model_files[1]['timestamp'] - self.model_files[0]['timestamp']
        return self.model_files[-1]['timestamp'] + timespan


class Indicators:
    def __init__(self, path, symbols):
        self.indicators = {}
        self.idx = {}
        self.next_timestamp = {}

        for symbol in symbols:
            for filename in os.listdir(path):
                if symbol in filename and filename[-4:] == '.csv':
                    file_path = os.path.join(path, filename)
                    break

            with open(file_path, 'rb') as f:
                self.indicators[symbol] = pd.read_csv(
                    filepath_or_buffer=f,
                    parse_dates=["timestamp", "ground_truth_timestamp"],
                    index_col='ind_idx')
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

    def set_start_date(self, start_date):
        for symbol in self.indicators:
            while self.indicators[symbol]['timestamp'].iloc[self.idx[symbol]] < start_date:
                self.idx[symbol] += 1

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
    step_count = 13
    training_path = 'E:/BitBot/simulation_data/'
    symbols = set()
    for filename in glob.glob(os.path.join(training_path, '*.csv')):
        symbols.add(filename.split('_')[-1].replace('.csv', ''))
    symbols = sorted(list(symbols))
    #symbols = symbols[0:1]

    profit_model = ProfitModel()
    indicators = Indicators(training_path, symbols)
    raw_predictions = {symbol: [] for symbol in symbols}
    raw_ground_truths = {symbol: [] for symbol in symbols}
    raw_timestamps = {symbol: [] for symbol in symbols}

    timestamp_start, timestamp_end = indicators.get_start_end_date()
    timestamp_end = min(timestamp_end, profit_model.last_predictable_timestamp())
    #timestamp_end = datetime(year=2020, month=7, day=15, tzinfo=timezone.utc)

    timestamp_start = profit_model.model_files[0]['timestamp']
    indicators.set_start_date(timestamp_start)

    timestamp = timestamp_start
    previous_timestamp = timestamp
    while timestamp < timestamp_end:
        if profit_model.has_new_model(timestamp):
            print(f"Predict {previous_timestamp} - {timestamp}")
            for symbol in symbols:
                # A section is all indicators that are covered by a single prediction model
                section = indicators.get_next_section(symbol, timestamp)
                if section.shape[0] == 0:
                    print(f"No data! {symbol} {timestamp}")
                    continue
                raw_predictions[symbol].append(profit_model.predict(section))
                raw_ground_truths[symbol].append(section.iloc[:, -step_count-1:-1].to_numpy())
                raw_timestamps[symbol].append(section.iloc[:, 0].to_numpy())
            profit_model.load_next_model()
            previous_timestamp = timestamp
        timestamp += timedelta(minutes=1)

    print(f"Predict {previous_timestamp} - {timestamp_end}")
    for symbol in symbols:
        section = indicators.get_next_section(symbol, timestamp_end)
        if section.shape[0] == 0:
            print(f"No data! {symbol} {timestamp}")
            continue
        raw_predictions[symbol].append(profit_model.predict(section))
        raw_ground_truths[symbol].append(section.iloc[:, -step_count-1:-1].to_numpy())
        raw_timestamps[symbol].append(section.iloc[:, 0].to_numpy())

    for symbol in symbols:
        raw_predictions[symbol] = np.concatenate(raw_predictions[symbol])
        raw_ground_truths[symbol] = np.concatenate(raw_ground_truths[symbol])
        raw_timestamps[symbol] = np.concatenate(raw_timestamps[symbol])

    predictions = raw_predictions
    ground_truths = raw_ground_truths
    timestamps = raw_timestamps

    """
    timestamps = []
    predictions = {symbol: [] for symbol in symbols}
    #prediction_indices = {symbol: [] for symbol in symbols}
    ground_truths = {symbol: [] for symbol in symbols}
    prediction_idx = {symbol: 0 for symbol in symbols}
    timestamp = timestamp_start
    timestamp_print = timestamp + timedelta(days=1)

    for symbol in symbols:
        while prediction_idx[symbol] < raw_predictions[symbol].shape[0] and timestamp >= indicators.indicators[symbol].iloc[prediction_idx[symbol]]['timestamp']:
            prediction_idx[symbol] += 1
        prediction_idx[symbol] = max(0, prediction_idx[symbol] - 1)

    while timestamp < timestamp_end:
        timestamps.append(timestamp)
        for symbol in symbols:
            #if prediction_idx[symbol] < raw_ground_truths[symbol].shape[0]:
            #else:

            if prediction_idx[symbol] < raw_predictions[symbol].shape[0] and timestamp >= indicators.indicators[symbol].iloc[prediction_idx[symbol]]['timestamp']:
                #prediction_indices[symbol].append(len(predictions[symbol]))
                predictions[symbol].append(raw_predictions[symbol][prediction_idx[symbol]])
                ground_truths[symbol].append(raw_ground_truths[symbol][prediction_idx[symbol]])
                prediction_idx[symbol] += 1
            else:
                ground_truths[symbol].append(None)
                predictions[symbol].append(None)

            if prediction_idx[symbol] < raw_predictions[symbol].shape[0] and timestamp >= indicators.indicators[symbol].iloc[prediction_idx[symbol]]['timestamp']:
                print("Error!")

        timestamp += timedelta(minutes=1)

        if timestamp >= timestamp_print:
            print(f"Compiling predictions {timestamp}")
            timestamp_print = timestamp + timedelta(days=1)
    """

    file_path = f"cache/predictions.pickle"
    with open(file_path, 'wb') as f:
        pickle.dump({
            'symbols': symbols,
            'timestamps': timestamps,
            'predictions': predictions,
            'ground_truths': ground_truths,
            #'prediction_indices': prediction_indices
        }, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    make_predictions()
