import os
import sys

import requests
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
from live_trader_logging import live_trader_setup_logging


class Portfolio:
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
        with open(f"portfolio.pickle", 'wb') as f:
            pickle.dump(self.positions, f)

    def load(self):
        try:
            with open(f"portfolio.pickle", 'rb') as f:
                self.positions = pickle.load(f)
        except FileNotFoundError:
            pass

    def __str__(self):
        s = "Portfolio:"
        for position in self.positions:
            s += f"\n  {position}"
        return s

"""
class Portfolios:
    portfolio_count = 20

    def __init__(self):
        self.portfolios = []
        for _ in range(self.portfolio_count):
            self.portfolios.append(Portfolio(self))
        self.load()

    def save(self):
        with open(f"portfolios.pickle", 'wb') as f:
            pickle.dump(self.portfolios, f)

    def load(self):
        try:
            with open(f"portfolios.pickle", 'rb') as f:
                self.portfolios = pickle.load(f)
        except FileNotFoundError:
            pass

    def __str__(self):
        s = "Portfolios:"
        #for position in self.positions:
        #    s += f"\n  {position}"
        return s
"""


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
    # Todo: spread out trades over each minute
    # Todo: circuit breaker
    # Todo: handle BNB fees
    # Todo: binance_account handle websocket errors https://github.com/sammchardy/python-binance/issues/834
    # Todo: long and short

    profit_model = load_learner('model_all_2021-07-01.pickle')

    with open('binance_account.json') as f:
        account_info = json.load(f)

    logging.info("Connect to Binance delta server")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://192.168.1.90:31007")

    nominal_order_size = 12.0  # usdt, slightly larger than min notional
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
    #portfolios = Portfolios()
    #for idx in range(20):
    #    portfolios.append(Portfolio())

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
        stri = f"Hodlings {total_equity_:.1f} USDT :"
        for h_symbol in symbols:
            balance = binance_account.get_balance(asset=h_symbol.replace('USDT', ''))
            if balance > 0:
                s_value = balance * binance_account.get_mark_price(symbol=h_symbol)
                stri += f" {s_value:.1f} {h_symbol}"
        logging.info(stri)
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
            #binance_account.sell_all()
            #quit()

        # check_positions(portfolio, binance_account)

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
        #for portfolio in portfolios.portfolios:
        for position in portfolio.positions.copy():
            mark_price = binance_account.get_mark_price(position['symbol'])
            if mark_price < position['stop_loss'] or mark_price > position['take_profit']:
                order_size = position['size']
                asset = position['symbol'].replace('USDT', '')
                account_balance = binance_account.get_balance(asset=asset)
                if order_size > account_balance or abs(order_size - account_balance) / account_balance < 0.1:
                    order_size = account_balance
                order_result = binance_account.market_sell(symbol=position['symbol'], volume=order_size)
                if order_result['quantity'] > 0:
                    logging.info(f"Sold {position['symbol']}: {order_size} @ {order_result['price']}, expected price: {mark_price}, {position}")
                    if position['size'] != order_result['quantity']:
                        logging.info(f"Executed quantity ({order_result['quantity']}) doesn't match position size ({position['size']})")
                    portfolio.remove_position(position)
                else:
                    logging.info(f"Sold {position['symbol']} FAILED: {order_size} @ {mark_price}, {position}")
                print_hodlings()

        # Buy
        # Predict values
        data_input = pd.DataFrame(data=directions, columns=directions_column_names)
        input_symbols = np.array(symbols)
        for symbol in symbols:
            data_input[symbol] = np.where(input_symbols == symbol, True, False)
        test_dl = profit_model.dls.test_dl(data_input)
        predictions = profit_model.get_preds(dl=test_dl)[0][:, 2].numpy() - 0.5

        position_max_count = min(100, int(binance_account.get_total_equity_usdt() / nominal_order_size))
        for _ in range(int(position_max_count)):
            # Todo: make chunks if the same symbol is bought multiple times

            cash, equity = binance_account.get_balance('USDT'), binance_account.get_total_equity_usdt(),
            if cash < equity / position_max_count * 0.9:
                break

            symbol_idx = random.randint(0, len(symbols) - 1)
            #random_symbol_order = random.sample(population=random_symbol_order, k=len(random_symbol_order))
            #for symbol_idx in random_symbol_order:
            prediction = predictions[symbol_idx]
            if prediction > 0 and 0.00020 * 5 * 25 ** prediction > random.random():
                symbol = symbols[symbol_idx]
                mark_price = binance_account.get_mark_price(symbol)

                order_value = min(equity / position_max_count, cash * 0.95)
                order_size = order_value / mark_price * 0.99

                #print(f"buy {kline_idx} {position_value} USDT {position_size:.2f} {symbols[symbol_idx]} @ {mark_price}")
                order_result = binance_account.market_buy(symbol=symbol, volume=order_size)
                if order_result['quantity'] > 0:
                    position = portfolio.add_position(symbol=symbol, position_size=order_result['quantity'], mark_price=order_result['price'])
                    logging.info(f"Bought {symbol} {order_result['quantity']} @ {order_result['price']} ({order_result['quantity'] * order_result['price']} USDT), {position}")
                    if order_result['quantity'] != order_size:
                        logging.info(f"Executed quantity ({order_result['quantity']}) doesn't match quoted quantity ({order_size})")
                else:
                    logging.info(f"Bought {symbol} FAILED {order_size} @ {order_result['price']}")
                print_hodlings()
                #break

        binance_account.update_balance()

        # Reset watchdog
        try:
            requests.get("http://localhost:31008/live_trader", timeout=5)
        except:
            pass


if __name__ == '__main__':
    live_trader_setup_logging(script_name=os.path.splitext(os.path.basename(sys.argv[0]))[0])

    #logging.basicConfig(
    #    format='%(asctime)s %(levelname)-8s %(message)s',
    #    level=logging.INFO,
    #    datefmt='%Y-%m-%d %H:%M:%S')
    #logging.Formatter.converter = time.gmtime
    main()
