import pickle
import random
import pandas as pd
from datetime import datetime, timezone, timedelta
from prediction_filter import PredictionFilter

from BinanceSimulator.binance_simulator import BinanceSimulator


class Portfolio:
    take_profit = 1.025
    stop_loss = 0.975

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
    file_path = f"cache/predictions.pickle"
    try:
        with open(file_path, 'rb') as f:
            prediction_data = pickle.load(f)
            symbols = prediction_data['symbols']
            predictions = prediction_data['predictions']
    except FileNotFoundError:
        print("Load predictions fail ", file_path)
        quit()

    # symbols = symbols[10:11]

    start_timestamp, end_timestamp = predictions[0]['timestamp'], predictions[-1]['timestamp']

    #start_timestamp = datetime.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    #end_timestamp = datetime.strptime("2021-06-15 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    simulator = BinanceSimulator(initial_usdt=10000, symbols=symbols)

    klines = {}
    for symbol in symbols:
        klines[symbol] = pd.read_hdf(f"cache/klines/{symbol}.hdf")

    start_timestamp_ms = start_timestamp.timestamp() * 1000
    kline_start_idx = {symbol: 0 for symbol in symbols}
    for symbol in symbols:
        kline_start_idx[symbol] = int((start_timestamp_ms - klines[symbol].iloc[0]['open_time']) / 60000)
        while klines[symbol].iloc[kline_start_idx[symbol]]['open_time'] > start_timestamp_ms:
            kline_start_idx[symbol] -= 1
        while klines[symbol].iloc[kline_start_idx[symbol]]['open_time'] < start_timestamp_ms:
            kline_start_idx[symbol] += 1

    #end_timestamp_ms = end_timestamp.timestamp() * 1000
    #kline_end_idx = int((end_timestamp_ms - klines[symbols[0]].iloc[0]['open_time']) / 60000)
    #kline_end_idx = min(kline_end_idx, klines[symbols[0]].shape[0])
    #while klines[symbols[0]].iloc[kline_end_idx]['open_time'] > end_timestamp_ms:
    #    kline_end_idx -= 1
    #while klines[symbols[0]].iloc[kline_end_idx]['open_time'] < end_timestamp_ms:
    #    kline_end_idx += 1

    #for symbol in symbols:
    #    print(symbol, klines[symbol].iloc[kline_start_idx]['open_time'], klines[symbol].iloc[kline_end_idx]['open_time'])

    #kline_start_idx = 31 * 24 * 60
    #kline_end_idx = klines[symbols[0]].shape[0] - 15 * 24 * 60

    #portfolio123 = hackattack.xsr.main();
    #config = portfolio123.run()
    portfolio = Portfolio()
    logger = Logger()
    previous_print = start_timestamp

    prediction_filters = {symbol: PredictionFilter() for symbol in symbols}

    def print_hodlings(_timestamp):
        nonlocal previous_print
        total_equity_ = simulator.get_equity_usdt()
        if _timestamp > previous_print + timedelta(days=1):
            previous_print += timedelta(hours=1)
            stri = f"{_timestamp} Hodlings {total_equity_:.1f} USDT"
            for w_symbol in simulator.wallet:
                if simulator.wallet[w_symbol] > 0:
                    s_value = simulator.wallet[w_symbol] * simulator.mark_price[w_symbol]
                    stri += f", {s_value:.1f} {w_symbol}"
            print(stri)
        logger.append(_timestamp, total_equity_, simulator.wallet['usdt'])

    """
    for symbol in symbols:
        ts_start = datetime.fromtimestamp(klines[symbol].iloc[0]['open_time'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        ts_end = datetime.fromtimestamp(klines[symbol].iloc[-1]['open_time'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        print(symbol, ts_start, ts_end)

    print("pred start", predictions[0]['timestamp'])
    print("pred end", predictions[-1]['timestamp'])
    """

    kline_idx = kline_start_idx.copy()
    for prediction_idx in range(len(predictions)):
        end_of_klines = False
        for symbol in symbols:
            while klines[symbol].iloc[kline_idx[symbol]]['open_time'] < predictions[prediction_idx]['timestamp'].timestamp() * 1000:
                kline_idx[symbol] += 1
                if kline_idx[symbol] >= klines[symbol].shape[0]:
                    end_of_klines
                    break

        if end_of_klines:
            break

        #for kline_idx in range(kline_start_idx, kline_end_idx):
        #prediction_idx = kline_idx - kline_start_idx

        for symbol in symbols:
            mark_price = klines[symbol]['close'][kline_idx[symbol]]
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
                print_hodlings(predictions[prediction_idx]['timestamp'])

        position_max_count = min(3, int(simulator.get_equity_usdt() / 12))

        # Buy
        buy_orders = {}
        cash, equity = simulator.get_equity_usdt(), simulator.get_cash_usdt()

        prediction_symbols = list(predictions[prediction_idx].keys())[1:]
        random.shuffle(prediction_symbols)

        position_count = len(portfolio.positions)

        for symbol in prediction_symbols:

            if position_count >= position_max_count:
                break

            # prediction_filters[symbol].append(predictions[prediction_idx][symbol])

            if simulator.wallet[symbol] > 0:
                continue

            p = predictions[prediction_idx][symbol][4]
            if 0.3 <= p <= 1.0:  # max(0.35, prediction_filters[symbol].smooth[-1]):  # 0.4:
                mark_price = klines[symbol]['close'][kline_idx[symbol]]
                simulator.set_mark_price(symbol=symbol, mark_price=mark_price)

                position_value = min(equity / position_max_count, cash * 0.98)
                if position_value < 10:
                    break
                position_size = position_value / mark_price * 0.99

                if symbol not in buy_orders:
                    buy_orders[symbol] = 0
                buy_orders[symbol] += position_size
                cash -= position_size * mark_price * (1 + 2 * 0.00075)
                equity -= position_size * mark_price * 2 * 0.00075
                position_count += 1

        """
        for _ in range(int(position_max_count)):
            if cash < equity / position_max_count * 0.9:
                break

            symbol_idx = random.randint(0, len(symbols) - 1)
            symbol = symbols[symbol_idx]

            #if prediction_idx is None:
            #    current_timestamp = datetime.fromtimestamp(klines[symbol]['open_time'][kline_idx] / 1000, tz=timezone.utc)
            #    timestamp_minutes = int((current_timestamp - start_timestamp).total_seconds() / 60)
            #    prediction_idx = timestamp_minutes

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
        """

        for symbol in buy_orders:
            position_size = buy_orders[symbol]
            mark_price = klines[symbol]['close'][kline_idx[symbol]]
            simulator.market_order(order_size=position_size, symbol=symbol)
            portfolio.add_position(symbol=symbol, position_size=position_size, mark_price=mark_price, kline_idx=kline_idx[symbol])
            print_hodlings(predictions[prediction_idx]['timestamp'])

    print_hodlings(predictions[prediction_idx - 1]['timestamp'])


if __name__ == '__main__':
    main()
