import pickle
import random

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from fastai.learner import load_learner

from BinanceSimulator.binance_simulator import BinanceSimulator


class Portfolio:
    position_max_count = 5
    take_profit = 1.05
    stop_loss = 0.95

    def __init__(self):
        self.positions = []

    def add_position(self, symbol: str, position_size: float, mark_price: float):
        self.positions.append({
            'symbol': symbol,
            'size': position_size,
            'mark_price': mark_price,
            'take_profit': mark_price * self.take_profit,
            'stop_loss': mark_price * self.stop_loss
        })

    def remove_position(self, position):
        self.positions.remove(position)


class Logger:
    def __init__(self):
        self.timestamps = []
        self.equities = []
        self.cash = []
        datestring = datetime.now().strftime("%Y-%m-%d %H%M%S")
        self.file = open(f"log/simlog {datestring}.txt", 'a')
        self.file.write(f"date,equity,cash\n")

    def append(self, timestamp, equity, cash):
        self.file.write(f"{timestamp},{equity},{cash}\n")
        self.file.flush()


def calculate_predictions(symbols, degrees, indicators, profit_model, kline_start_idx, kline_end_idx):
    file_path = f"cache/tmp_predictions.pickle"
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        pass

    first_symbol = list(indicators.keys())[0]
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
    predictions = np.empty((kline_end_idx - kline_start_idx, len(symbols)))

    for kline_idx in range(kline_start_idx, kline_end_idx):
        for symbol_idx, symbol in enumerate(symbols):
            tmp_indicator_columns[symbol_idx] = indicators[symbol]['indicators'][:, :, kline_idx].transpose().flatten()
        df_indicators = pd.DataFrame(data=tmp_indicator_columns, columns=indicator_column_names)
        df = pd.concat([df_symbols, df_indicators], axis=1)

        test_dl = profit_model.dls.test_dl(df)
        predictions[kline_idx - kline_start_idx] = profit_model.get_preds(dl=test_dl)[0][:, 2].numpy() - 0.5
        if kline_idx % 100 == 0:
            print(f"Computing predictions {kline_idx / kline_end_idx * 100:.2f}%, {kline_idx} / {kline_end_idx}")

    with open(file_path, 'wb') as f:
        pickle.dump(predictions, f)

    return predictions


def main():
    with open(f"cache/filtered_symbols.pickle", 'rb') as f:
        symbols = pickle.load(f)

    klines = {}
    indicators = {}
    for symbol in symbols:
        with open(f"cache/indicators/{symbol}.pickle", 'rb') as f:
            indicators[symbol] = pickle.load(f)
        klines[symbol] = pd.read_hdf(f"cache/klines/{symbol}.hdf")

    profit_model = load_learner('model_all_2021-06-23.pickle')

    #with open(f"cache/intrinsic_events.pickle", 'rb') as f:
    #    intrinsic_events = pickle.load(f)

    kline_start_idx = (31 + 14) * 24 * 60
    kline_end_idx = None
    for symbol in symbols:
        data_length_symbol = klines[symbol]['close'].shape[0]
        if kline_end_idx is None:
            kline_end_idx = data_length_symbol
        else:
            kline_end_idx = min(kline_end_idx, data_length_symbol)
    #mark_price = klines[symbol]['close'][kline_idx]
    #kline_end_idx = indicators[list(indicators.keys())[0]]['indicators'].shape[2]
    # start_timestamp is start of kline data
    start_timestamp = datetime.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_timestamp = datetime.strptime("2021-06-15 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    simulator = BinanceSimulator(initial_usdt=1000, symbols=symbols)

    degrees = [1, 2, 3]

    predictions = calculate_predictions(
        symbols,
        degrees,
        indicators,
        profit_model,
        kline_start_idx,
        kline_end_idx
    )

    random_symbol_order = list(range(len(symbols)))

    #portfolio = Portfolio()
    portfolios = []
    for idx in range(20):
        portfolios.append(Portfolio())

    logger = Logger()

    def print_hodlings(kline_idx_):
        timestamp = start_timestamp + timedelta(minutes=kline_idx_)
        total_equity_ = simulator.get_equity_usdt()
        stri = f"{timestamp} Hodlings {total_equity_:.1f} USDT"
        for w_symbol in simulator.wallet:
            if simulator.wallet[w_symbol] > 0:
                s_value = simulator.wallet[w_symbol] * simulator.mark_price[w_symbol]
                stri += f", {s_value:.1f} {w_symbol}"
        print(stri)
        logger.append(timestamp, total_equity_, simulator.wallet['usdt'])

    for kline_idx in range(kline_start_idx, kline_end_idx):
        for symbol in symbols:
            mark_price = klines[symbol]['close'][kline_idx]
            simulator.set_mark_price(symbol=symbol, mark_price=mark_price)

        for portfolio in portfolios:
            # Sell
            for position in portfolio.positions[:]:
                mark_price = simulator.mark_price[position['symbol']]
                if mark_price < position['stop_loss'] or mark_price > position['take_profit']:
                    order_size = -position['size']
                    if abs(position['size'] - simulator.wallet[position['symbol']]) / simulator.wallet[position['symbol']] < 0.1:
                        order_size = -simulator.wallet[position['symbol']]
                    simulator.market_order(order_size=order_size, symbol=position['symbol'])
                    portfolio.remove_position(position)
                    print_hodlings(kline_idx)

        # Buy
        equity = simulator.get_equity_usdt()
        cash = simulator.get_cash_usdt()
        if cash > equity / (len(portfolios) * portfolios[0].position_max_count):
            #for symbol_idx, symbol in enumerate(symbols):
            #    tmp_indicator_columns[symbol_idx] = indicators[symbol]['indicators'][:, :, kline_idx].transpose().flatten()

            #df_indicators = pd.DataFrame(data=tmp_indicator_columns, columns=indicator_column_names)
            #df = pd.concat([df_symbols, df_indicators], axis=1)

            #test_dl = profit_model.dls.test_dl(df)
            #predictions = profit_model.get_preds(dl=test_dl)[0][:, 2].numpy() - 0.5

            random_symbol_order = random.sample(population=random_symbol_order, k=len(random_symbol_order))

            for portfolio in portfolios:
                k = 0.00020
                for symbol_idx in random_symbol_order:
                    prediction = predictions[kline_idx - kline_start_idx][symbol_idx]
                    if prediction > 0:  # and predictions_ema50[idx] > 0:
                        t = k * 25 ** prediction
                        r = random.random()
                        if r < t:
                            mark_price = klines[symbols[symbol_idx]]['close'][kline_idx]
                            simulator.set_mark_price(symbol=symbols[symbol_idx], mark_price=mark_price)

                            position_value = min(equity / (len(portfolios) * portfolio.position_max_count), cash)
                            position_size = position_value / mark_price * 0.98

                            simulator.market_order(order_size=position_size, symbol=symbols[symbol_idx])
                            #print(f"buy {kline_idx} {position_value} USDT {position_size:.2f} {symbols[symbol_idx]} @ {mark_price}")

                            portfolio.add_position(symbol=symbols[symbol_idx], position_size=position_size, mark_price=mark_price)
                            print_hodlings(kline_idx)
                            break

            #print(predictions)

    print_hodlings(kline_end_idx - 1)


if __name__ == '__main__':
    main()
