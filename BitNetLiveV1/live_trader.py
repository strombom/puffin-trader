
import os
import sys
import zmq
import time
import json
import random
import pickle
import logging
import pathlib
import requests
import threading
import itertools
import collections
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from fastai.learner import load_learner

from IntrinsicTime.runner import Runner
#from binance_account import BinanceAccount
from live_trader_logging import live_trader_setup_logging
from bybit_account import BybitAccount


class Portfolio:
    take_profit = 1.03
    stop_loss = 0.97

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

        if binance_account.get_balance(symbol=symbol) > used[symbol] * 0.9:
            new_positions.append(position)

    portfolio.positions = new_positions


class IntrinsicEvents:
    def __init__(self, lengths):
        self.step_count = lengths[-1]
        self.runners = {}
        self.steps = {}

    def run(self, prices, symbols, deltas):
        symbols_with_new_steps = set()
        for price in prices:
            for symbol in symbols:
                if symbol not in self.runners:
                    self.runners[symbol] = Runner(delta=deltas[symbol])
                    self.steps[symbol] = collections.deque(maxlen=self.step_count)
                runner_steps = self.runners[symbol].step(price[symbol])
                for step in runner_steps:
                    self.steps[symbol].append(step)
                    symbols_with_new_steps.add(symbol)
        return symbols_with_new_steps, self.steps


class Indicators:
    def __init__(self):
        self.direction_degrees = [1, 2, 3]
        self.lengths = [5, 7, 11, 15, 22, 33, 47, 68, 100, 150]

    def make(self, symbols, steps):
        directions = {symbol: np.empty((len(self.direction_degrees) * len(self.lengths))) for symbol in symbols}
        price_diffs = {symbol: np.empty((len(self.direction_degrees) * len(self.lengths))) for symbol in symbols}
        for symbol in symbols:
            for direction_degree_idx, direction_degree in enumerate(self.direction_degrees):
                for length_idx, length in enumerate(self.lengths):
                    idx = self.lengths[-1]
                    start, end = idx - length, idx
                    xp = np.arange(start, end)
                    direction_steps = list(itertools.islice(steps[symbol], start, end))
                    yp = np.poly1d(np.polyfit(xp, direction_steps, direction_degree))
                    curve = yp(xp)
                    direction = curve[-1] / curve[-2] - 1.0
                    price_diff = curve[-1] / direction_steps[-1] - 1.0
                    directions[symbol][direction_degree_idx * len(self.lengths) + length_idx] = direction
                    price_diffs[symbol][direction_degree_idx * len(self.lengths) + length_idx] = price_diff
        return directions, price_diffs


class ProfitModel:
    def __init__(self, direction_degrees, lengths, base_path):
        self.file_path = os.path.join(base_path, 'models/model.pickle')

        self.base_path = base_path
        self.model = None
        self.model_creation_timestamp = None
        self.deltas = {}
        self.load_model()

        self.directions_column_names = []
        self.price_diffs_column_names = []
        for _, direction_degree in enumerate(direction_degrees):
            for _, length in enumerate(lengths):
                self.directions_column_names.append(f"{direction_degree}-{length}-d")
                self.price_diffs_column_names.append(f"{direction_degree}-{length}-p")

    def get_model_creation_timestamp(self):
        return datetime.fromtimestamp(pathlib.Path(self.file_path).stat().st_mtime)

    def load_deltas(self):
        deltas_df = pd.read_csv(os.path.join(self.base_path, 'deltas.csv'))
        for idx, row in deltas_df.iterrows():
            self.deltas[row['symbol']] = row['delta']

    def load_model(self):
        try:
            self.model = load_learner(self.file_path)
            self.model_creation_timestamp = self.get_model_creation_timestamp()
            logging.info(f"Loaded new model {self.model_creation_timestamp}")
        except:
            logging.info("Failed to load new model")
        try:
            self.load_deltas()
        except:
            logging.info("Failed to load deltas")

    def predict(self, all_symbols, symbols_with_new_steps, directions, price_diffs):
        data_input = pd.DataFrame(data=np.vstack(list(directions.values())), columns=self.directions_column_names)
        data_input[self.price_diffs_column_names] = np.vstack(list(price_diffs.values()))

        deltas_array = []
        input_symbols = np.array(list(symbols_with_new_steps))
        for symbol_idx, symbol in enumerate(all_symbols):
            data_input[symbol] = np.where(input_symbols == symbol, True, False)
            if symbol in symbols_with_new_steps:
                deltas_array.append(self.deltas[symbol])
        data_input['delta'] = deltas_array

        new_timestamp = self.get_model_creation_timestamp()
        if new_timestamp > self.model_creation_timestamp:
            self.load_model()

        test_dl = self.model.dls.test_dl(data_input)
        predictions_array = self.model.get_preds(dl=test_dl)[0]
        predictions_array = predictions_array[:, 5].numpy()

        predictions = {}
        for symbol_idx, symbol in enumerate(symbols_with_new_steps):
            predictions[symbol] = predictions_array[symbol_idx]

        logging.info(f"Predictions: {predictions}")

        return predictions


class PriceClient:
    def __init__(self):
        logging.info("Connect to Binance delta server")
        self._last_prices_idx = 0
        self.context = None
        self.socket = None
        self.connect()

    def connect(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.AFFINITY, 1)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.socket.connect("tcp://superdator.se:31007")

    def disconnect(self):
        self.socket.close()
        self.context.term()

    def get_new_prices(self):
        requesting = True
        while requesting:
            try:
                command, payload = 'get_since', self._last_prices_idx
                self.socket.send_pyobj((command, payload))
                message = self.socket.recv_pyobj()
                if message['last_idx'] != self._last_prices_idx:
                    self._last_prices_idx = message['last_idx']
                    if 'prices' in message and len(message['prices']) > 0:
                        requesting = False
            except zmq.error.Again:
                self.disconnect()
                self.connect()
            if not requesting:
                break
            time.sleep(1)

        prices = message['prices']
        logging.info("New prices", prices[-1])
        return prices


def test_bench():
    #pc = PriceClient()
    #prices = pc.get_new_prices()

    base_path = 'C:/BitBotLiveV1'

    indicators = Indicators()
    intrinsic_events = IntrinsicEvents(lengths=indicators.lengths)

    prices = pd.read_csv(f"{base_path}/test_klines.csv")
    all_symbols = list(prices.columns)[1:]
    prices = prices.to_dict('records')

    profit_model = ProfitModel(direction_degrees=indicators.direction_degrees, lengths=indicators.lengths, base_path=base_path)
    profit_model.load_deltas()

    # Intrinsic events
    symbols_with_new_steps, steps = intrinsic_events.run(prices=prices, symbols=all_symbols, deltas=profit_model.deltas)

    # Make indicators
    directions, price_diffs = indicators.make(symbols=symbols_with_new_steps, steps=steps)

    # Make predictions
    predictions = profit_model.predict(
        all_symbols=all_symbols,
        symbols_with_new_steps=symbols_with_new_steps,
        directions=directions,
        price_diffs=price_diffs
    )

    print(predictions)


class HodlingsPrinter:
    def __init__(self, bybit_account, all_symbols, logger):
        self._logger = logger
        self._all_symbols = all_symbols
        self._bybit_account = bybit_account
        self._old_balance = {key: 0 for key in all_symbols}

    def print_hodlings(self):
        changed = False
        timestamp = datetime.now(tz=timezone.utc)
        total_equity = self._bybit_account.get_total_equity_usdt()
        logstr = f"Hodlings {total_equity:.1f} USDT :"
        for symbol in self._all_symbols:
            balance = self._bybit_account.get_balance(symbol=symbol)
            if balance != self._old_balance[symbol]:
                changed = True
                self._old_balance[symbol] = balance
            if balance > 0:
                s_value = balance * self._bybit_account.get_mark_price(symbol=symbol)
                logstr += f" {s_value:.1f} {symbol}"

        if changed:
            logging.info(logstr)
            self._logger.append(timestamp, total_equity)


def main():
    # Todo: spread out trades over each minute
    # Todo: circuit breaker
    # Todo: handle BNB fees
    # Todo: binance_account handle websocket errors https://github.com/sammchardy/python-binance/issues/834
    # Todo: long and short

    nominal_order_size = 10.0 + 2.0  # usdt, slightly larger than min notional

    price_client = PriceClient()
    indicators = Indicators()
    intrinsic_events = IntrinsicEvents(lengths=indicators.lengths)
    profit_model = ProfitModel(direction_degrees=indicators.direction_degrees, lengths=indicators.lengths, base_path="C:/BitBotLiveV1/")
    portfolio = Portfolio()
    logger = Logger()
    all_symbols = []
    bybit_account = None
    hodlings_printer = None

    while True:
        # Get latest price data
        prices = price_client.get_new_prices()

        # Initialise variables
        if len(all_symbols) == 0:
            # Symbols must exist in profit model
            all_symbols = sorted([symbol for symbol in prices[0].keys() if symbol in profit_model.deltas])
            if len(all_symbols) != len(profit_model.deltas):
                logging.error("Bitnet kline server does not have all profit model prices!")
                return
            with open('credentials.json') as f:
                credentials = json.load(f)
            bybit_account = BybitAccount(
                api_key=credentials['bybit']['api_key'],
                api_secret=credentials['bybit']['api_secret'],
                symbols=all_symbols
            )
            hodlings_printer = HodlingsPrinter(bybit_account=bybit_account, all_symbols=all_symbols, logger=logger)
            #binance_account.sell_all()
            #quit()

        # check_positions(portfolio, binance_account)

        # Intrinsic events
        symbols_with_new_steps, steps = intrinsic_events.run(prices=prices, symbols=all_symbols, deltas=profit_model.deltas)

        # Make indicators
        directions, price_diffs = indicators.make(symbols=symbols_with_new_steps, steps=steps)

        # Sell
        for position in portfolio.positions.copy():
            mark_price = bybit_account.get_mark_price(position['symbol'])
            if mark_price < position['stop_loss'] or mark_price > position['take_profit']:
                order_size = position['size']
                position_balance = bybit_account.get_balance(symbol=position['symbol'])
                if order_size > position_balance or abs(order_size - position_balance) / position_balance < 0.1:
                    order_size = position_balance
                order_result = bybit_account.market_order(symbol=position['symbol'], volume=-order_size)
                if order_result['quantity'] > 0:
                    logging.info(f"Sold {position['symbol']}: {order_size} @ {order_result['price']}, expected price: {mark_price}, {position}")
                    if position['size'] != order_result['quantity']:
                        logging.info(f"Executed quantity ({order_result['quantity']}) doesn't match position size ({position['size']})")
                    portfolio.remove_position(position)
                else:
                    if order_result['error'] == 'low volume':
                        logging.info(f"Sold {position['symbol']} FAILED (low volume), removed position: {order_size} @ {mark_price}, {position}")
                        portfolio.remove_position(position)
                    else:
                        logging.info(f"Sold {position['symbol']} FAILED: {order_size} @ {mark_price}, {position}")

        # Buy
        if len(symbols_with_new_steps) > 0:
            predictions = profit_model.predict(
                all_symbols=all_symbols,
                symbols_with_new_steps=symbols_with_new_steps,
                directions=directions,
                price_diffs=price_diffs
            )

            position_max_count = min(3, int(bybit_account.get_total_equity_usdt() / nominal_order_size))

            prediction_symbols = list(symbols_with_new_steps)
            random.shuffle(prediction_symbols)

            for symbol in prediction_symbols:
                if not bybit_account.has_symbol(symbol):
                    continue
                if len(portfolio.positions) >= position_max_count:
                    break

                # Todo: make chunks if the same symbol is bought multiple times

                cash, equity = bybit_account.get_balance('USDT'), bybit_account.get_total_equity_usdt(),
                if cash < equity / position_max_count * 0.9:
                    break

                prediction = predictions[symbol]
                if 0.3 <= prediction <= 1.0:
                    mark_price = bybit_account.get_mark_price(symbol)

                    order_value = min(equity / position_max_count, cash * 0.975)
                    order_size = order_value / mark_price * 0.99

                    #print(f"buy {kline_idx} {position_value} USDT {position_size:.2f} {symbols[symbol_idx]} @ {mark_price}")
                    order_result = bybit_account.market_order(symbol=symbol, volume=order_size)
                    if order_result['quantity'] > 0:
                        position = portfolio.add_position(symbol=symbol, position_size=order_result['quantity'], mark_price=order_result['price'])
                        logging.info(f"Bought {symbol} {order_result['quantity']} @ {order_result['price']} ({order_result['quantity'] * order_result['price']} USDT), {position}")
                        if order_result['quantity'] != order_size:
                            logging.info(f"Executed quantity ({order_result['quantity']}) doesn't match quoted quantity ({order_size})")
                    else:
                        logging.info(f"Bought {symbol} FAILED {order_size} @ {order_result['price']}")
                    #break

        bybit_account.update_balance()
        if hodlings_printer:
            hodlings_printer.print_hodlings()

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
    #test_bench()
