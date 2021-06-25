
import zmq
import time
import json
import random
import pickle
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
        self.load()

    def add_position(self, symbol: str, position_size: float, mark_price: float):
        position = {
            'symbol': symbol,
            'size': position_size,
            'mark_price': mark_price,
            'take_profit': mark_price * self.take_profit,
            'stop_loss': mark_price * self.stop_loss
        }
        self.positions.append(position)
        self.save()
        return position

    def remove_position(self, position):
        self.positions.remove(position)
        self.save()

    def save(self):
        with open(f"position.pickle", 'wb') as f:
            pickle.dump(self.positions, f)

    def load(self):
        try:
            with open(f"position.pickle", 'rb') as f:
                self.positions = pickle.load(f)
        except FileNotFoundError:
            pass

    def __str__(self):
        s = "Portfolio:"
        for position in self.positions:
            s += f"\n  {position}"
        return s


class Logger:
    def __init__(self):
        self.timestamps = []
        self.equities = []
        self.file = open(f"log.txt", 'a')

    def append(self, timestamp, equity):
        self.file.write(f"{timestamp},{equity}\n")
        self.file.flush()


def check_positions(portfolio, binance_account):
    # Todo: Add missing positions
    # Todo: Modify position sizes
    used = {}
    new_positions = []
    for position in portfolio.positions:
        symbol = position['symbol']
        if symbol not in used:
            used[symbol] = 0
        used[symbol] += position['size']

        if binance_account.get_balance(asset=symbol.replace('USDT', '')) > used[symbol] * 0.9:
            new_positions.append(position)

    portfolio.positions = new_positions


def main():
    profit_model = load_learner('model_all_2021-06-23.pickle')

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

    random_symbol_order = None
    portfolio = Portfolio()
    print(portfolio)
    logger = Logger()

    directions = None
    binance_account = None

    directions_column_names = []
    for _, direction_degree in enumerate(direction_degrees):
        for _, length in enumerate(lengths):
            directions_column_names.append(f"{direction_degree}-{length}")

    def print_hodlings():
        timestamp = datetime.now(tz=timezone.utc)
        total_equity_ = binance_account.get_total_equity_usdt()
        stri = f"{timestamp} Hodlings {total_equity_:.1f} USDT :"
        for h_symbol in symbols:
            balance = binance_account.get_balance(asset=h_symbol.replace('USDT', ''))
            if balance > 0:
                s_value = balance * binance_account.get_mark_price(symbol=h_symbol)
                stri += f" {s_value:.1f} {h_symbol}"
        print(stri)
        logger.append(timestamp, total_equity_)

    while True:
        # Get latest price data
        command, payload = 'get_since', last_data_idx
        socket.send_pyobj((command, payload))
        message = socket.recv_pyobj()
        if message['last_idx'] == last_data_idx:
            time.sleep(1)
            continue
        last_data_idx = message['last_idx']
        prices = message['prices']
        logging.info("New prices")

        # Initialise variables
        if len(symbols) == 0:
            symbols = sorted(prices[0].keys())
            random_symbol_order = list(range(len(symbols)))
            directions = np.empty((len(symbols), len(direction_degrees) * len(lengths)))
            binance_account = BinanceAccount(
                api_key=account_info['api_key'],
                api_secret=account_info['api_secret'],
                symbols=symbols
            )
            # binance_account.sell_all()

        check_positions(portfolio, binance_account)

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
            mark_price = binance_account.get_mark_price(position['symbol'])
            if mark_price < position['stop_loss'] or mark_price > position['take_profit']:
                order_size = position['size']
                asset = position['symbol'].replace('USDT', '')
                account_balance = binance_account.get_balance(asset=asset)
                if order_size > account_balance or abs(order_size - account_balance) / account_balance < 0.1:
                    order_size = account_balance
                if binance_account.market_sell(symbol=position['symbol'], volume=order_size):
                    logging.info(f"Sold position {order_size} @ {mark_price}, {position}")
                    portfolio.remove_position(position)
                else:
                    logging.info(f"Sold position FAILED {order_size} @ {mark_price}, {position}")
                print_hodlings()

        # Buy
        equity = binance_account.get_total_equity_usdt()
        cash = binance_account.get_balance('USDT')
        if cash > equity / portfolio.position_max_count * 0.25:
            # Predict values
            data_input = pd.DataFrame(data=directions, columns=directions_column_names)
            input_symbols = np.array(symbols)
            for symbol in symbols:
                data_input[symbol] = np.where(input_symbols == symbol, True, False)
            test_dl = profit_model.dls.test_dl(data_input)
            predictions = profit_model.get_preds(dl=test_dl)[0][:, 2].numpy() - 0.5

            random_symbol_order = random.sample(population=random_symbol_order, k=len(random_symbol_order))

            k = 0.00020
            for symbol_idx in random_symbol_order:
                prediction = predictions[symbol_idx]
                if prediction > 0:  # and predictions_ema50[idx] > 0:
                    t = k * 25 ** prediction
                    r = random.random()
                    if r < t:
                        symbol = symbols[symbol_idx]
                        mark_price = binance_account.get_mark_price(symbol)

                        position_value = min(equity / portfolio.position_max_count, cash)
                        position_size = position_value / mark_price * 0.98

                        #print(f"buy {kline_idx} {position_value} USDT {position_size:.2f} {symbols[symbol_idx]} @ {mark_price}")
                        if binance_account.market_buy(symbol=symbol, volume=position_size):
                            position = portfolio.add_position(symbol=symbol, position_size=position_size, mark_price=mark_price)
                            logging.info(f"Bought position {position_size} @ {mark_price}, {position}")
                        else:
                            logging.info(f"Bought position FAILED {position_size} @ {mark_price}, {position}")
                        print_hodlings()
                        break


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.Formatter.converter = time.gmtime
    main()
