import pickle
import random
import pandas as pd
from datetime import datetime, timezone, timedelta
#from prediction_filter import PredictionFilter

from BinanceSimulator.binance_simulator import BinanceSimulator


class Portfolio:
    take_profit = 1.018
    stop_loss = 0.982

    def __init__(self, capacity):
        self.capacity = capacity
        self.positions = [None] * self.capacity

    #def has_symbol(self, symbol):
    #    for position in self.positions:
    #        if position is not None:
    #            if symbol == position['symbol']:
    #                return True
    #    return False

    def count_active_positions(self):
        return self.capacity - self.positions.count(None)

    def set_position(self, idx, symbol: str, position_size: float, mark_price: float, kline_idx: int):
        self.positions[idx] = {
            'idx': idx,
            'kline_idx': kline_idx,
            'symbol': symbol,
            'size': position_size,
            'mark_price': mark_price,
            'take_profit': mark_price * self.take_profit,
            'stop_loss': mark_price * self.stop_loss
        }

    def has_position(self, idx):
        return self.positions[idx] is not None

    def add_position(self, symbol, position_size, mark_price, kline_idx):
        position_idx = -1
        for idx, position in enumerate(self.positions):
            if position is None:
                position_idx = idx
                break
        if position_idx == -1:
            print("Error! No position available!")
            quit()

        self.set_position(
            idx=position_idx,
            symbol=symbol,
            position_size=position_size,
            mark_price=mark_price,
            kline_idx=kline_idx
        )

    def update_limits(self, position, mark_price):
        position['take_profit'] = mark_price * self.take_profit,
        position['stop_loss'] = mark_price * self.stop_loss,

    def get_lowest_score_position(self):
        if None in self.positions:
            return None

        min_idx, min_score = -1, 1

        for idx, position in enumerate(self.positions):
            if position is None:
                continue
            if position['score'] < min_score:
                min_idx, min_score = idx, position['score']

        if min_idx >= 0:
            return self.positions[min_idx]
        return None


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


class CountLog:
    def __init__(self):
        date_string = datetime.now().strftime("%Y-%m-%d %H%M%S")
        self.file = open(f"log/countlog_{date_string}.txt", 'a')
        self.file.write(f"timestamp,count\n")

    def append(self, timestamp, count):
        self.file.write(f"{timestamp},{count}\n")
        self.file.flush()


def sig_fig_round(number, digits):
    x = float(f'%.{digits}g' % number)
    return x

    #factor = 1 / self._tick_size[symbol]
    #quantity = math.floor(abs(volume) * factor) / factor
    #return quantity

    #power = "{:e}".format(number).split('e')[1]
    #return round(number, -(int(power) - digits - 1))


def main():
    file_path = f"cache/predictions.pickle"
    try:
        with open(file_path, 'rb') as f:
            prediction_data = pickle.load(f)
            symbols = prediction_data['symbols']
            timestamps = prediction_data['timestamps']
            predictions = prediction_data['predictions']
            #prediction_indices = prediction_data['prediction_indices']
            ground_truths = prediction_data['ground_truths']
    except FileNotFoundError:
        print("Load predictions fail ", file_path)
        quit()

    thresholds = [0.225, 0.25, 0.275, 0.3]
    thresholds = [0.275]
    # symbols = symbols[10:11]
    symbols = ["ADAUSDT", "BCHUSDT", "BNBUSDT", "BTTUSDT", "CHZUSDT", "EOSUSDT", "ETCUSDT", "LINKUSDT", "MATICUSDT",
               "THETAUSDT", "XLMUSDT", "XRPUSDT"]
    #symbols = ["MATICUSDT"]

    #start_timestamp, end_timestamp = timestamps[0], timestamps[-1]

    timestamp_start = datetime.strptime("2020-07-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    timestamp_end = datetime.strptime("2021-10-17 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    simulator = BinanceSimulator(initial_usdt=10000, symbols=symbols)

    klines = {}
    for symbol in symbols:
        klines[symbol] = pd.read_hdf(f"cache/klines/{symbol}.hdf")
2
    timestamp_start_ms = timestamp_start.timestamp() * 1000
    kline_start_idx = {symbol: 0 for symbol in symbols}
    for symbol in symbols:
        kline_start_idx[symbol] = 264118
        continue

        kline_start_idx[symbol] = int((timestamp_start_ms - klines[symbol].iloc[0]['open_time']) / 60000)
        while klines[symbol].iloc[kline_start_idx[symbol]]['open_time'] > timestamp_start_ms:
            kline_start_idx[symbol] -= 1
        while klines[symbol].iloc[kline_start_idx[symbol]]['open_time'] < timestamp_start_ms:
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
    portfolio = Portfolio(capacity=len(symbols) * len(thresholds))
    #portfolio_ghost = Portfolio(5)
    logger = Logger()
    previous_print = timestamp_start

    #prediction_filters = {symbol: PredictionFilter() for symbol in symbols}

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

    #cool_off_period = timedelta(hours=1)
    #cool_off = predictions[0]['timestamp'] - cool_off_period * 2

    def sell(position):
        symbol = position['symbol']
        mark_price = klines[symbol]['close'][kline_idx[symbol]]
        #simulator.set_mark_price(symbol=symbol, mark_price=mark_price)
        order_size = -position['size']
        if abs(position['size'] - simulator.wallet[symbol]) / simulator.wallet[symbol] < 0.1:
            order_size = -simulator.wallet[symbol]
        simulator.market_order(order_size=order_size, symbol=symbol)
        portfolio.positions[position['idx']] = None
        print_hodlings(timestamps[symbol][prediction_idx[symbol]])

    count_log = CountLog()

    prediction_idx = {symbol: 0 for symbol in symbols}
    kline_idx = kline_start_idx.copy()
    timestamp = timestamp_start
    # for timestamp_idx in range(0, len(timestamps)):
    while timestamp < timestamp_end:
        end_of_klines = False
        for symbol in symbols:
            while klines[symbol].iloc[kline_idx[symbol]]['open_time'] < timestamp.timestamp() * 1000:
                kline_idx[symbol] += 1
                if kline_idx[symbol] >= klines[symbol].shape[0]:
                    end_of_klines = True
                    break
        if end_of_klines:
            break

        for symbol in symbols:  # simulator.get_positions_symbols():
            mark_price = klines[symbol]['close'][kline_idx[symbol]]
            simulator.set_mark_price(symbol=symbol, mark_price=mark_price)

        prediction_symbols = []
        for symbol in symbols:
            while timestamps[symbol][prediction_idx[symbol]] < timestamp:
                prediction_idx[symbol] += 1

            if timestamps[symbol][prediction_idx[symbol]] == timestamp:
                prediction_symbols.append(symbol)
        #prediction_symbols = list(predictions[prediction_idx].keys())[1:]
        random.shuffle(prediction_symbols)

        #for symbol in prediction_symbols:
        #    prediction_score = predictions[prediction_idx][symbol][3]
        #    if prediction_score <= -0.25:
        #        for position in portfolio.positions:
        #            if position is not None and position['symbol'] == symbol:
        #                sell(position)

        #for kline_idx in range(kline_start_idx, kline_end_idx):
        #prediction_idx = kline_idx - kline_start_idx

        # Sell
        #for position in portfolio_ghost.positions:
        #    if position is None:
        #        continue
        #    mark_price = simulator.mark_price[position['symbol']]
        #    if mark_price < position['stop_loss'] or mark_price > position['take_profit']:
        #        portfolio_ghost.positions[position['idx']] = None
        #        #print("ghost_portfolio_count", portfolio_ghost.count_active_positions())
        #        count_log.append(timestamps[timestamp_idx], portfolio_ghost.count_active_positions())

        for position in portfolio.positions:
            if position is None:
                continue
            mark_price = simulator.mark_price[position['symbol']]
            if mark_price < position['stop_loss'] or mark_price > position['take_profit']:  # or cool_off > predictions[prediction_idx]['timestamp']:
                #if cool_off > predictions[prediction_idx]['timestamp']:
                #    print(predictions[prediction_idx]['timestamp'], "Cool off sell", position)
                sell(position)

        #position_max_count = min(100, int(simulator.get_equity_usdt() / 12))

        #if cool_off > predictions[prediction_idx]['timestamp']:
        #    continue

        # Buy
        #buy_orders = {}
        #cash, equity = simulator.get_equity_usdt(), simulator.get_cash_usdt()

        #print(predictions[prediction_idx]['timestamp'])

        #position_count = len(portfolio.positions)

        #for symbol in prediction_symbols:
        #    if None not in portfolio_ghost.positions:
        #        break
        #    prediction_score = predictions[symbol][timestamp_idx][6]
        #    if prediction_score > 0.35:
        #        mark_price = klines[symbol]['close'][kline_idx[symbol]]
        #        portfolio_ghost.add_position(symbol=symbol, position_size=1, mark_price=mark_price,
        #                                     kline_idx=kline_idx[symbol], score=prediction_score)
        #        count_log.append(timestamps[timestamp_idx], portfolio_ghost.count_active_positions())

        #ghost_portfolio_count = portfolio_ghost.count_active_positions()

        for symbol in prediction_symbols:
            #if position_count >= position_max_count:
            #    break

            # prediction_filters[symbol].append(predictions[prediction_idx][symbol])

            #if simulator.wallet[symbol] > 0:
            #    continue

            #if ghost_portfolio_count < 3:
            #    break

            #threshold = thresholds[symbol][timestamp_idx]
            prediction_score = predictions[symbol][prediction_idx[symbol]][8]
            #if 0.6 < prediction_score < 1.0:  # max(0.35, prediction_filters[symbol].smooth[-1]):  # 0.4:

            for threshold_idx, threshold in enumerate(thresholds):
                if prediction_score > threshold:

                    #if position_count == position_max_count - 1:
                    #    print(predictions[prediction_idx]['timestamp'], "Start cool off")
                    #    cool_off = predictions[prediction_idx]['timestamp'] + cool_off_period

                    #lowest_score_position = portfolio.get_lowest_score_position()
                    #if lowest_score_position is not None:
                    #    if prediction_score < lowest_score_position['score']:
                    #        #print(f"No slot for {symbol}, score {prediction_score}")
                    #        continue
                    #    if symbol == lowest_score_position['symbol']:
                    #        portfolio.update_limits(lowest_score_position, mark_price)
                    #        continue
                    #    else:
                    #        sell(lowest_score_position)

                    position_idx = symbols.index(symbol) * len(thresholds) + threshold_idx

                    if portfolio.has_position(position_idx):
                        continue

                    #if None not in portfolio.positions:
                    #    continue

                    #if portfolio.has_symbol(symbol):
                    #    continue

                    #    print("oj")
                    #simulator.set_mark_price(symbol=symbol, mark_price=mark_price)

                    #simulator.set_mark_price(symbol=symbol, mark_price=mark_price)
                    equity, cash = simulator.get_equity_usdt(), simulator.get_cash_usdt()
                    position_value = min(equity / portfolio.capacity, cash * 0.98)
                    if position_value < 10:
                        break
                    mark_price = klines[symbol]['close'][kline_idx[symbol]]
                    position_size = sig_fig_round(position_value / mark_price * 0.99, 5)

                    simulator.market_order(order_size=position_size, symbol=symbol)

                    portfolio.set_position(
                        idx=position_idx,
                        symbol=symbol,
                        position_size=position_size,
                        mark_price=mark_price,
                        kline_idx=kline_idx[symbol]
                    )

                    #portfolio.add_position(symbol=symbol, position_size=position_size, mark_price=mark_price,
                    #                       kline_idx=kline_idx[symbol])
                    print_hodlings(timestamps[symbol][prediction_idx[symbol]])

                    #if symbol not in buy_orders:
                    #    buy_orders[symbol] = 0
                    #buy_orders[symbol] += position_size
                    #cash -= position_size * mark_price * (1 + 2 * 0.00075)
                    #equity -= position_size * mark_price * 2 * 0.00075
                    #position_count += 1

        timestamp += timedelta(minutes=1)
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

        """
        for symbol in buy_orders:
            position_size = buy_orders[symbol]
            mark_price = klines[symbol]['close'][kline_idx[symbol]]
            simulator.market_order(order_size=position_size, symbol=symbol)
            portfolio.add_position(symbol=symbol, position_size=position_size, mark_price=mark_price, kline_idx=kline_idx[symbol])
            print_hodlings(predictions[prediction_idx]['timestamp'])
        """

    print_hodlings(timestamps[timestamp_idx - 1])


if __name__ == '__main__':
    main()
