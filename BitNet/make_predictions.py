import os
import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from fastai.learner import load_learner


def calculate_predictions(indicators):
    start_timestamp = datetime.now()

    #first_symbol = list(indicators.keys())[0]
    #data_length = indicators[first_symbol]['indicators'].shape[2]
    #lengths = indicators[first_symbol]['lengths']
    #indicator_column_names = []
    #for degree in degrees:
    #    for length in lengths:
    #        indicator_column_names.append(f"{degree}-{length}")

    #tmp_symbol_columns = np.empty((len(symbols), len(symbols)), dtype=bool)
    #tmp_symbol_columns.fill(False)
    #np.fill_diagonal(tmp_symbol_columns, True)
    #df_symbols = pd.DataFrame(tmp_symbol_columns, columns=symbols)

    #tmp_indicator_columns = np.empty((len(symbols), len(degrees) * len(indicators[first_symbol]['lengths'])))
    #predictions = np.empty((data_length, len(symbols)))

    symbols = set(indicators.keys())

    model_files = []
    for filename in glob.glob('model_all_*_*.pickle'):
        model_files.append({
            'timestamp': datetime.strptime(filename[21:31], "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(hours=6),
            'filename': filename
        })
    model_files = sorted(model_files, key=lambda x: x['timestamp'])

    model_idx = 0
    profit_model = load_learner(model_files[model_idx]['filename'])

    start_date, end_date = None, None
    for symbol in symbols:
        first_date = indicators[symbol].iloc[0]['timestamp']
        last_date = indicators[symbol].iloc[-1]['timestamp']
        if start_date is None:
            start_date = first_date
        if end_date is None:
            end_date = last_date

        start_date = max(start_date, first_date)
        end_date = max(end_date, last_date)

    """
    predictions = {}
    for symbol in symbols:
        predictions[symbol] = []
        indicator = indicators[symbol].drop(columns=['timestamp'])
        indicator = indicator.rename(columns={'1-5-p': 'timestamp"1-5-p'})

        for idx in range(indicator.shape[0]):
            print(idx)

        test_dl = profit_model.dls.test_dl(indicator)
        predictions[symbol].append(profit_model.get_preds(dl=test_dl)[0][:, 0].numpy() - 0.5)
    """

    indicator_idx = {}
    indicator_next_timestamp = {}
    for symbol in symbols:
        indicator_idx[symbol] = 0
        indicator_next_timestamp[symbol] = indicators[symbol].iloc[0]['timestamp']

    predictions = {}
    for symbol in symbols:
        predictions[symbol] = []

    timestamp = start_date
    while timestamp < end_date:

        # Load current model
        if model_idx + 1 < len(model_files):
            if timestamp >= model_files[model_idx + 1]['timestamp']:

                for symbol in symbols:
                    start_idx = indicator_idx[symbol]
                    end_idx = start_idx
                    while end_idx + 1 < indicators[symbol].shape[0]:
                        if indicators[symbol]['timestamp'].iloc[end_idx + 1] >= timestamp:
                            break
                        end_idx += 1

                    print(f"Predict {symbol} {start_idx}-{end_idx}")

                    indicator_section = indicators[symbol].iloc[start_idx:end_idx]
                    indicator_section = indicator_section.rename(columns={'1-5-p': 'timestamp"1-5-p'})

                    test_dl = profit_model.dls.test_dl(indicator_section)
                    predictions[symbol].append(profit_model.get_preds(dl=test_dl)[0][:, 0].numpy())

                    indicator_idx[symbol] = end_idx

                model_idx += 1
                profit_model = load_learner((model_files[model_idx]['filename']))

        """
        prediction = {}
        
        for symbol in symbols:
            if timestamp == indicator_next_timestamp[symbol]:
                indicator = indicators[symbol].iloc[indicator_idx[symbol]]

                if indicator_idx[symbol] + 1 < indicators[symbol].shape[0]:
                    indicator_idx[symbol] += 1
                    indicator_next_timestamp[symbol] = indicators[symbol].iloc[0]['date']

        predictions.append(prediction)
        """

        timestamp += timedelta(minutes=1)

    for symbol in symbols:
        predictions[symbol] = np.concatenate(predictions[symbol])

    data_length = (end_date - start_date).minutes
    predictions = []

    print(data_length)

    #predictions = np.empty((all_indicators.shape[0], len(indicators)))



    #all_indicators = []
    #for symbol in indicators:
    #    start_idx = int(indicators[symbol].shape[0] * 0.85)
    #    all_indicators.append(indicators[symbol][start_idx:])
    #all_indicators = pd.concat(all_indicators)

    #predictions = np.empty((all_indicators.shape[0], len(indicators)))

    for idx in range(data_length):
        #for symbol_idx, symbol in enumerate(symbols):
        #    tmp_indicator_columns[symbol_idx] = indicators[symbol]['indicators'][:, :, idx].transpose().flatten()
        #df_indicators = pd.DataFrame(data=tmp_indicator_columns, columns=indicator_column_names)
        #df = pd.concat([df_symbols, df_indicators], axis=1)

        test_dl = profit_model.dls.test_dl(indicators[1])
        predictions[idx] = profit_model.get_preds(dl=test_dl)[0][:, 2].numpy() - 0.5
        if idx % 100 == 0 and idx > 0:
            current_timestamp = datetime.now()
            elapsed = current_timestamp - start_timestamp
            predicted_end = start_timestamp + elapsed / (idx / data_length)

            print(f"Computing predictions {idx / data_length * 100:.2f}%, {idx} / {data_length}, {predicted_end}")

    file_path = f"cache/predictions.pickle"
    with open(file_path, 'wb') as f:
        pickle.dump(predictions, f)


def make_predictions():
    training_path = 'E:/BitBot/training_data/'
    symbols = set()
    for filename in os.listdir(training_path):
        symbols.add(filename.replace('.csv', ''))
        break
    symbols = list(symbols)

    #with open(f"cache/filtered_symbols.pickle", 'rb') as f:
    #    symbols = pickle.load(f)

    degrees = [1, 2, 3]

    indicators = {}
    for symbol in symbols:
        with open(f"E:/BitBot/training_data/{symbol}.csv", 'rb') as f:
            indicators[symbol] = pd.read_csv(f, parse_dates=[0])

    calculate_predictions(indicators)


if __name__ == '__main__':
    make_predictions()
