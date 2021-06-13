import zmq
import json
import threading
import collections
import pyrate_limiter
from datetime import datetime, timedelta
from binance import Client, ThreadedWebsocketManager


class Klines:
    def __init__(self, symbol_count: int):
        self.symbol_count = symbol_count

        self.last_idx = 0

        self.history_klines = []
        self.history_current_kline = {}

        self.buffer = collections.deque(maxlen=30 * 24 * 60)
        self.buffer_current_kline = {}

        self.lock = threading.Lock()

    def append(self, symbol: str, price: float):
        with self.lock:
            self.buffer_current_kline[symbol] = price
            if len(self.buffer_current_kline) == self.symbol_count:
                self.buffer.append(self.buffer_current_kline)
                self.last_idx += 1
                self.buffer_current_kline.clear()

    def history_append(self, symbol: str, price: float):
        self.history_current_kline[symbol] = price
        if len(self.history_current_kline) == self.symbol_count:
            self.history_klines.append(self.history_current_kline)
            self.history_current_kline = {}

    def history_finish(self):
        with self.lock:
            for kline in reversed(self.history_klines):
                self.buffer.appendleft(kline)
                if len(self.buffer) == self.buffer.maxlen:
                    break


class BinanceKlines:
    def __init__(self, klines: Klines, binance_client: Client, start_time: datetime, symbols: list):
        self.mark_prices = {}

        self._klines = klines
        self._client = binance_client
        self._assets = {}
        self._kline_threads = {}
        self.symbols = symbols

        #self.check_symbols()

        def process_kline_message(data):
            if data['e'] == 'kline':
                self.mark_prices[data['s']] = float(data['k']['c'])

        for symbol in self.symbols:
            kline_socket_manager = BinanceSocketManager(self._client)
            kline_socket_manager.start_kline_socket(symbol=symbol, callback=process_kline_message, interval='1m')
            kline_socket_manager.start()
            self._kline_threads[symbol] = kline_socket_manager

        self.download_history(start_time)

        from time import sleep
        has_all_symbols = False
        while not has_all_symbols:
            has_all_symbols = True
            for symbol in self.symbols:
                if symbol + "USDT" not in self.mark_prices:
                    has_all_symbols = False
                    sleep(0.2)
                    break

    def check_symbols(self):
        all_symbols = set()
        exchange_info = self._client.get_exchange_info()
        for symbol in exchange_info['symbols']:
            all_symbols.add(symbol['symbol'])

        for symbol in self.symbols:
            if symbol not in all_symbols:
                print(f"BinanceKlines, Error {symbol} is not available!")
                quit()

    def download_history(self, start_time: datetime):
        limiter = pyrate_limiter.Limiter(pyrate_limiter.RequestRate(1, pyrate_limiter.Duration.SECOND))

        @limiter.ratelimit("download_print_status", delay=False)
        def print_status(timestamp):
            print(f"{datetime.now()} {timestamp}")

        for symbol in self.symbols:
            for kline in self._client.get_historical_klines_generator(
                    symbol,
                    Client.KLINE_INTERVAL_1MINUTE,
                    start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            ):
                close_price = float(kline[4])
                self._klines.history_append(symbol, close_price)

        self._klines.history_finish()


def server():
    symbols = [
        'ADAUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'BTTUSDT', 'CHZUSDT', 'DOGEUSDT', 'EOSUSDT', 'ETCUSDT', 'ETHUSDT',
        'LINKUSDT', 'LTCUSDT', 'MATICUSDT', 'NEOUSDT', 'THETAUSDT', 'TRXUSDT', 'VETUSDT', 'XLMUSDT', 'XRPUSDT'
    ]
    symbols = ['ADAUSDT', 'BCHUSDT']

    klines = Klines(symbol_count=len(symbols))

    with open('binance_account.json') as f:
        account_info = json.load(f)

    start_time = datetime.utcnow() - timedelta(minutes=3)  # 30 * 24 * 60)

    client = Client(
        api_key=account_info['api_key'],
        api_secret=account_info['api_secret']
    )

    binance_klines = BinanceKlines(klines=klines, binance_client=client, start_time=start_time, symbols=symbols)

    """
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:31007")

    while True:
        command, payload = socket.recv_pyobj()
        send_data = None

        with history_lock:
            if command == 'get_all':
                send_data = {
                    'last_idx': history_counter,
                    'mark_prices': history
                }

            elif command == 'get_since':
                last_idx = payload
                history_idx = len(history) - 1 - (history_counter - last_idx)
                history_idx = max(0, history_idx)
                send_data = {
                    'last_idx': history_counter,
                    'mark_prices': list(itertools.islice(history, history_idx, len(history)))
                }

        socket.send_pyobj(send_data)
    """


if __name__ == "__main__":
    server()
