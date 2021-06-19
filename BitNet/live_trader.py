
import zmq
import time
import json
import logging
import itertools
import collections
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from fastai.learner import load_learner

from IntrinsicTime.runner import Runner
from binance_account import BinanceAccount


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


def main():
    profit_model = load_learner('model_all.pickle')

    with open('binance_account.json') as f:
        account_info = json.load(f)

    logging.info("Connect to Binance delta server")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:31007")

    delta = 0.01
    direction_degrees = [1, 2, 3]
    lengths = pd.read_csv('cache/regime_data_lengths.csv')['length'].to_list()
    step_count = lengths[-1]

    # indicators = None
    last_data_idx = 0
    symbols = []
    steps = {}
    runners = {}
    portfolio = Portfolio()

    directions = None
    binance_account = None

    directions_column_names = []
    for _, direction_degree in enumerate(direction_degrees):
        for _, length in enumerate(lengths):
            directions_column_names.append(f"{direction_degree}-{length}")

    def print_hodlings(kline_idx_):
        timestamp = datetime.now(tz=timezone.utc)
        total_equity_ = binance_account.get_total_equity_usdt()
        stri = f"{timestamp} Hodlings {total_equity_:.1f} USDT"
        for w_symbol in simulator.wallet:
            if simulator.wallet[w_symbol] > 0:
                s_value = simulator.wallet[w_symbol] * simulator.mark_price[w_symbol]
                stri += f", {s_value:.1f} {w_symbol}"
        print(stri)
        logger.append(timestamp, total_equity_)

    while True:
        # Get latest price data
        command, payload = 'get_since', last_data_idx
        socket.send_pyobj((command, payload))
        message = socket.recv_pyobj()
        if message['last_idx'] == last_data_idx:
            time.sleep(secs=1)
            continue
        last_data_idx = message['last_idx']
        prices = message['prices']

        # Initialise variables
        if len(symbols) == 0:
            symbols = sorted(prices[0].keys())
            directions = np.empty((len(symbols), len(direction_degrees) * len(lengths)))
            binance_account = BinanceAccount(
                api_key=account_info['api_key'],
                api_secret=account_info['api_secret'],
                symbols=symbols
            )

        # Runners
        for price in prices:
            for symbol in symbols:
                if symbol not in runners:
                    runners[symbol] = Runner(delta=delta)
                    steps[symbol] = collections.deque(maxlen=step_count)
                runner_steps = runners[symbol].step(price[symbol])
                for step in runner_steps:
                    steps[symbol].append(step)

        # Make directions
        for symbol_idx, symbol in enumerate(symbols):
            for direction_degree_idx, direction_degree in enumerate(direction_degrees):
                for length_idx, length in enumerate(lengths):
                    idx = lengths[-1]
                    start, end = idx - length, idx
                    xp = np.arange(start, end)
                    direction_steps = list(itertools.islice(steps[symbol], start, end))
                    yp = np.poly1d(np.polyfit(xp, direction_steps, direction_degree))
                    curve = yp(xp)
                    direction = curve[-1] / curve[-2] - 1.0
                    directions[symbol_idx, direction_degree_idx * len(lengths) + length_idx] = direction

        # Sell
        for position in portfolio.positions[:]:
            # TODO: Sell
            mark_price = binance_account.get_mark_price(position['symbol'])
            if mark_price < position['stop_loss'] or mark_price > position['take_profit']:
                order_size = -position['size']
                if abs(position['size'] - simulator.wallet[position['symbol']]) / simulator.wallet[position['symbol']] < 0.1:
                    order_size = -simulator.wallet[position['symbol']]
                simulator.market_order(order_size=order_size, symbol=position['symbol'])
                portfolio.positions.remove(position)
                print_hodlings(kline_idx)

        # Predict values
        data_input = pd.DataFrame(data=directions, columns=directions_column_names)
        input_symbols = np.array(symbols)
        for symbol in symbols:
            data_input[symbol] = np.where(input_symbols == symbol, True, False)
        test_dl = profit_model.dls.test_dl(data_input)
        predictions = profit_model.get_preds(dl=test_dl)[0][:, 2].numpy() - 0.5

        equity = binance_account.get_total_equity_usdt()
        cash = binance_account.get_balance('USDT')
        if cash > equity / portfolio.position_max_count:
            for symbol_idx, symbol in enumerate(symbols):
                tmp_indicator_columns[symbol_idx] = indicators[symbol]['indicators'][:, :, kline_idx].transpose().flatten()

            df_indicators = pd.DataFrame(data=tmp_indicator_columns, columns=indicator_column_names)
            df = pd.concat([df_symbols, df_indicators], axis=1)

            test_dl = profit_model.dls.test_dl(df)
            predictions = profit_model.get_preds(dl=test_dl)[0][:, 2].numpy() - 0.5

            random_symbol_order = random.sample(population=random_symbol_order, k=len(random_symbol_order))

            k = 0.00020
            for symbol_idx in random_symbol_order:
                prediction = predictions[symbol_idx]
                if prediction > 0:  # and predictions_ema50[idx] > 0:
                    t = k * 25 ** prediction
                    r = random.random()
                    if r < t:
                        mark_price = klines[symbols[symbol_idx]]['close'][kline_idx]
                        simulator.set_mark_price(symbol=symbols[symbol_idx], mark_price=mark_price)

                        position_value = min(equity / portfolio.position_max_count, cash)
                        position_size = position_value / mark_price * 0.98

                        simulator.market_order(order_size=position_size, symbol=symbols[symbol_idx])
                        #print(f"buy {kline_idx} {position_value} USDT {position_size:.2f} {symbols[symbol_idx]} @ {mark_price}")

                        portfolio.add_position(symbol=symbols[symbol_idx], position_size=position_size, mark_price=mark_price)
                        print_hodlings(kline_idx)
                        break


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.Formatter.converter = time.gmtime
    main()
