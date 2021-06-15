import zmq
import json
import itertools
import threading
import traceback
import collections
import pyrate_limiter
from datetime import datetime, timedelta, timezone
from binance import Client, ThreadedWebsocketManager


class MinutePriceBuffer:
    def __init__(self, symbol_count: int):
        self.symbol_count = symbol_count

        # Only used while downloading historical klines
        self.last_idx = 0
        self.history_prices = {}

        # Price buffer
        self.buffer_prices = collections.deque(maxlen=30 * 24 * 60)

        self.lock = threading.Lock()

    def get_all(self):
        with self.lock:
            last_idx, prices = self.last_idx, list(self.buffer_prices)
            return last_idx, prices

    def get_since(self, since_idx):
        with self.lock:
            start_idx = len(self.buffer_prices) - (self.last_idx - since_idx)
            start_idx = max(0, start_idx)
            prices = []
            if start_idx < len(self.buffer_prices):
                prices = list(itertools.islice(self.buffer_prices, start_idx, None))
            return self.last_idx, prices

    def append(self, prices: dict):
        # Append live data
        with self.lock:
            self.buffer_prices.append(prices)
            self.last_idx += 1

    def history_append(self, symbol: str, price: float):
        # Create history
        if symbol not in self.history_prices:
            self.history_prices[symbol] = []
        self.history_prices[symbol].append(price)

    def history_finish(self):
        # Check that all historical prices have the same length
        lengths = [len(self.history_prices[symbol]) for symbol in self.history_prices]
        if not all(x == lengths[0] for x in lengths):
            print("Error! Not all historical prices have the same length!")
            quit()

        # Add history to buffer
        for idx in range(lengths[0] - 1, 0, -1):
            prices = {}
            for symbol in self.history_prices:
                prices[symbol] = self.history_prices[symbol][idx]
            with self.lock:
                if len(self.buffer_prices) == self.buffer_prices.maxlen:
                    return
                self.buffer_prices.appendleft(prices)
                self.last_idx += 1


class BinanceKlines:
    def __init__(self, klines: MinutePriceBuffer, start_time: datetime, symbols: list):
        self._klines = klines
        self.symbols = symbols
        self.current_prices = {}

        history_end_time = datetime.now(timezone.utc)
        self.next_timestamp = history_end_time + timedelta(minutes=1)

        with open('binance_account.json') as f:
            account_info = json.load(f)

        self._client = Client(
            api_key=account_info['api_key'],
            api_secret=account_info['api_secret']
        )

        self.check_symbols()

        def process_tick_message(data):
            symbol = data['data']['s']
            last_price = float(data['data']['p'])
            self.current_prices[symbol] = last_price

            current_timestamp = datetime.now(timezone.utc)
            if current_timestamp > self.next_timestamp and len(self.current_prices) == len(self.symbols):
                print("Appending", self.next_timestamp, self.current_prices)
                self.next_timestamp = self.next_timestamp + timedelta(minutes=1)
                self._klines.append(self.current_prices.copy())

        self.twm = ThreadedWebsocketManager(
            api_key=account_info['api_key'],
            api_secret=account_info['api_secret']
        )
        self.twm.start()

        streams = [f"{symbol.lower()}@trade" for symbol in self.symbols]
        self.twm.start_multiplex_socket(callback=process_tick_message, streams=streams)

        self.download_history(start_time=start_time, end_time=history_end_time)

    def check_symbols(self):
        all_symbols = set()
        exchange_info = self._client.get_exchange_info()
        for symbol in exchange_info['symbols']:
            all_symbols.add(symbol['symbol'])

        for symbol in self.symbols:
            if symbol not in all_symbols:
                print(f"BinanceKlines, Error {symbol} is not available!")
                quit()

    def download_history(self, start_time: datetime, end_time: datetime):
        #limiter = pyrate_limiter.Limiter(pyrate_limiter.RequestRate(1, pyrate_limiter.Duration.SECOND))
        #@limiter.ratelimit("download_print_status", delay=False)
        #def print_status(timestamp):
        #    print(f"{datetime.now()} {timestamp}")

        for symbol in self.symbols:
            start_timestamp = int(datetime.timestamp(start_time) * 1000)
            end_timestamp = int(datetime.timestamp(end_time) * 1000)

            for kline in self._client.get_historical_klines_generator(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_1MINUTE,
                    start_str=start_timestamp,
                    end_str=end_timestamp
            ):
                close_price = float(kline[4])
                self._klines.history_append(symbol, close_price)

        self._klines.history_finish()


def server():
    symbols = [
        'ADAUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'BTTUSDT', 'CHZUSDT', 'DOGEUSDT', 'EOSUSDT', 'ETCUSDT', 'ETHUSDT',
        'LINKUSDT', 'LTCUSDT', 'MATICUSDT', 'NEOUSDT', 'THETAUSDT', 'TRXUSDT', 'VETUSDT', 'XLMUSDT', 'XRPUSDT'
    ]
    symbols = ['BTCUSDT', 'BNBUSDT']

    klines = MinutePriceBuffer(symbol_count=len(symbols))

    history_start_time = datetime.now(timezone.utc) - timedelta(minutes=1 * 24 * 60)  # 30 * 24 * 60)

    print("Downloading history")

    _binance_klines = BinanceKlines(klines=klines, start_time=history_start_time, symbols=symbols)

    print("History downloaded")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:31007")

    while True:
        try:
            command, payload = socket.recv_pyobj()
        except:
            print("Receive pyobj error!")
            traceback.print_exc()
            continue

        send_data = None

        if command == 'get_all':
            last_idx, prices = klines.get_all()
            send_data = {
                'last_idx': last_idx,
                'prices': prices
            }

        elif command == 'get_since':
            last_idx = payload
            last_idx, prices = klines.get_since(last_idx)
            send_data = {
                'last_idx': last_idx,
                'prices': prices
            }

        socket.send_pyobj(send_data)


if __name__ == "__main__":
    server()
