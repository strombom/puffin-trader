import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastai.learner import load_learner


def calculate_predictions(symbols, degrees, indicators, profit_model):
    start_timestamp = datetime.now()

    first_symbol = list(indicators.keys())[0]
    data_length = indicators[first_symbol]['indicators'].shape[2]
    lengths = indicators[first_symbol]['lengths']
    indicator_column_names = []
    for degree in degrees:
        for length in lengths:
            indicator_column_names.append(f"{degree}-{length}")

    tmp_symbol_columns = np.empty((len(symbols), len(symbols)), dtype=bool)
    tmp_symbol_columns.fill(False)
    np.fill_diagonal(tmp_symbol_columns, True)
    df_symbols = pd.DataFrame(tmp_symbol_columns, columns=symbols)

    tmp_indicator_columns = np.empty((len(symbols), len(degrees) * len(indicators[first_symbol]['lengths'])))
    predictions = np.empty((data_length, len(symbols)))

    for idx in range(data_length):
        for symbol_idx, symbol in enumerate(symbols):
            tmp_indicator_columns[symbol_idx] = indicators[symbol]['indicators'][:, :, idx].transpose().flatten()
        df_indicators = pd.DataFrame(data=tmp_indicator_columns, columns=indicator_column_names)
        df = pd.concat([df_symbols, df_indicators], axis=1)

        test_dl = profit_model.dls.test_dl(df)
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
    with open(f"cache/filtered_symbols.pickle", 'rb') as f:
        symbols = pickle.load(f)

    degrees = [1, 2, 3]

    indicators = {}
    for symbol in symbols:
        with open(f"cache/indicators/{symbol}.pickle", 'rb') as f:
            indicators[symbol] = pickle.load(f)

    profit_model = load_learner('model_all_2021-07-02.pickle')

    calculate_predictions(
        symbols,
        degrees,
        indicators,
        profit_model
    )


if __name__ == '__main__':
    make_predictions()
