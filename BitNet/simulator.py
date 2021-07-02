import pickle
import random

import pandas as pd
from datetime import datetime, timezone, timedelta

from BinanceSimulator.binance_simulator import BinanceSimulator


class Portfolio:
    take_profit = 1.05
    stop_loss = 0.95

    def __init__(self):
        self.positions = []

    def add_position(self, symbol: str, position_size: float, mark_price: float, kline_idx: int):
        self.positions.append({
            'kline_idx': kline_idx,
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
        date_string = datetime.now().strftime("%Y-%m-%d %H%M%S")
        self.file = open(f"log/simlog {date_string}.txt", 'a')
        self.file.write(f"date,equity,cash\n")

    def append(self, timestamp, equity, cash):
        self.file.write(f"{timestamp},{equity},{cash}\n")
        self.file.flush()


def main():
    with open(f"cache/filtered_symbols.pickle", 'rb') as f:
        symbols = pickle.load(f)

    file_path = f"cache/tmp_predictions.pickle"
    try:
        with open(file_path, 'rb') as f:
            predictions = pickle.load(f)
    except FileNotFoundError:
        print("Load predictions fail ", file_path)
        quit()

    start_timestamp = datetime.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_timestamp = datetime.strptime("2021-06-15 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    simulator = BinanceSimulator(initial_usdt=10000, symbols=symbols)

    klines = {}
    for symbol in symbols:
        klines[symbol] = pd.read_hdf(f"cache/klines/{symbol}.hdf")

    kline_start_idx = 31 * 24 * 60
    kline_end_idx = klines[symbols[0]].shape[0] - 15 * 24 * 60

    portfolio = Portfolio()
    logger = Logger()
    previous_print = start_timestamp

    def print_hodlings(kline_idx_):
        nonlocal previous_print

        total_equity_ = simulator.get_equity_usdt()
        timestamp = start_timestamp + timedelta(minutes=kline_idx_)
        if timestamp > previous_print + timedelta(days=1):
            previous_print += timedelta(days=1)
            stri = f"{timestamp} Hodlings {total_equity_:.1f} USDT"
            for w_symbol in simulator.wallet:
                if simulator.wallet[w_symbol] > 0:
                    s_value = simulator.wallet[w_symbol] * simulator.mark_price[w_symbol]
                    stri += f", {s_value:.1f} {w_symbol}"
            print(stri)
        logger.append(timestamp, total_equity_, simulator.wallet['usdt'])

    for kline_idx in range(kline_start_idx, kline_end_idx):
        prediction_idx = None

        for symbol in symbols:
            mark_price = klines[symbol]['close'][kline_idx]
            simulator.set_mark_price(symbol=symbol, mark_price=mark_price)

        # Sell
        for position in portfolio.positions.copy():
            mark_price = simulator.mark_price[position['symbol']]
            if mark_price < position['stop_loss'] or mark_price > position['take_profit']:
                order_size = -position['size']
                if abs(position['size'] - simulator.wallet[position['symbol']]) / simulator.wallet[position['symbol']] < 0.1:
                    order_size = -simulator.wallet[position['symbol']]
                simulator.market_order(order_size=order_size, symbol=position['symbol'])
                portfolio.remove_position(position)
                print_hodlings(kline_idx)

        position_max_count = min(50, int(simulator.get_equity_usdt() / 12))

        # Buy
        buy_orders = {}
        cash, equity = simulator.get_equity_usdt(), simulator.get_cash_usdt()
        for _ in range(int(position_max_count)):
            if cash < equity / position_max_count * 0.9:
                break

            symbol_idx = random.randint(0, len(symbols) - 1)
            symbol = symbols[symbol_idx]

            if prediction_idx is None:
                current_timestamp = datetime.fromtimestamp(klines[symbol]['open_time'][kline_idx] / 1000, tz=timezone.utc)
                timestamp_minutes = int((current_timestamp - start_timestamp).total_seconds() / 60)
                prediction_idx = timestamp_minutes

            prediction = predictions[prediction_idx][symbol_idx]
            if prediction > 0 and 0.00020 * 10 * 25 ** prediction > random.random():

                mark_price = klines[symbol]['close'][kline_idx]
                simulator.set_mark_price(symbol=symbol, mark_price=mark_price)

                position_value = min(equity / position_max_count, cash * 0.95)
                if position_value < 10:
                    break
                position_size = position_value / mark_price * 0.99

                if symbol not in buy_orders:
                    buy_orders[symbol] = 0
                buy_orders[symbol] += position_size
                cash -= position_size * mark_price * (1 + 2 * 0.00075)
                equity -= position_size * mark_price * 2 * 0.00075
                #break

        for symbol in buy_orders:
            position_size = buy_orders[symbol]
            mark_price = klines[symbol]['close'][kline_idx]
            simulator.market_order(order_size=position_size, symbol=symbol)
            portfolio.add_position(symbol=symbol, position_size=position_size, mark_price=mark_price, kline_idx=kline_idx)
            print_hodlings(kline_idx)

    print_hodlings(kline_end_idx - 1)


if __name__ == '__main__':
    main()
