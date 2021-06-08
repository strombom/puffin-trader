import pickle
import random

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from fastai.learner import load_learner

from BinanceSimulator.binance_simulator import BinanceSimulator


class Portfolio:
    def __init__(self):
        self.cash = 1000
        self.positions = []

    def has_cash(self):
        return True


def main():
    with open(f"cache/filtered_symbols.pickle", 'rb') as f:
        symbols = pickle.load(f)

    klines = {}
    indicators = {}
    for symbol in symbols:
        with open(f"cache/indicators/{symbol}.pickle", 'rb') as f:
            indicators[symbol] = pickle.load(f)
        klines[symbol] = pd.read_hdf(f"cache/klines/{symbol}.hdf")

    profit_model = load_learner('model_all.pickle')

    #with open(f"cache/intrinsic_events.pickle", 'rb') as f:
    #    intrinsic_events = pickle.load(f)

    data_length = indicators[list(indicators.keys())[0]]['indicators'].shape[2]
    start_timestamp = datetime.strptime("2020-02-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_timestamp = datetime.strptime("2021-05-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    kline_start_idx = 14 * 24 * 60

    simulator = BinanceSimulator(initial_usdt=1000, symbols=symbols)

    take_profit = 1.05
    stop_loss = 0.95

    first_symbol = list(indicators.keys())[0]
    lengths = indicators[first_symbol]['lengths']
    degrees = [1, 2, 3]

    indicator_column_names = []
    for degree in degrees:
        for length in lengths:
            indicator_column_names.append(f"{degree}-{length}")
    tmp_indicator_columns = np.empty((len(symbols), len(degrees) * len(indicators[first_symbol]['lengths'])))

    tmp_symbol_columns = np.empty((len(symbols), len(symbols)), dtype=bool)
    tmp_symbol_columns.fill(False)
    np.fill_diagonal(tmp_symbol_columns, True)
    df_symbols = pd.DataFrame(tmp_symbol_columns, columns=symbols)

    random_symbol_order = list(range(len(symbols)))

    portfolio = Portfolio()

    for kline_idx in range(kline_start_idx, data_length):
        for position in portfolio.positions:
            print(position)

        if portfolio.has_cash():
            for symbol_idx, symbol in enumerate(symbols):
                tmp_indicator_columns[symbol_idx] = indicators[symbol]['indicators'][:, :, kline_idx].transpose().flatten()

            df_indicators = pd.DataFrame(data=tmp_indicator_columns, columns=indicator_column_names)
            df = pd.concat([df_symbols, df_indicators], axis=1)

            test_dl = profit_model.dls.test_dl(df)
            predictions = profit_model.get_preds(dl=test_dl)[0][:, 2].numpy() - 0.5

            random_symbol_order = random.sample(population=random_symbol_order, k=len(random_symbol_order))

            k = 0.001
            for symbol_idx in random_symbol_order:
                prediction = predictions[symbol_idx]
                if prediction > 0:  # and predictions_ema50[idx] > 0:
                    t = k * 100 ** prediction
                    r = random.random()
                    if r < t:
                        print("buy", kline_idx, symbols[symbol_idx])

            #print(predictions)


if __name__ == '__main__':
    main()
