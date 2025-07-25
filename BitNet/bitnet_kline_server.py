import asyncio

import zmq
import hmac
import time
import json
import itertools
import threading
import traceback
import websockets
import collections
import pyrate_limiter
from datetime import datetime, timedelta, timezone
from binance import Client



"""
class BybitWebsocket:
    def __init__(self, api_key: str, api_secret: str):
        self._api_key = api_key
        self._api_secret = api_secret
        self._uri_private = "wss://stream.bybit.com/realtime_private"
        self._uri_public = "wss://stream.bybit.com/realtime_public"
        self._websockets = []

    def position(self, callback):
        connect_promise = self._websocket_connect(
            uri=self._uri_private,
            topics=["position"],
            callback=callback,
            authenticate=True
        )
        position_websocket = threading.Thread(target=asyncio.run, args=(connect_promise,))
        position_websocket.start()
        self._websockets.append(position_websocket)

    def trades(self, symbols, callback):
        connect_promise = self._websocket_connect(
            uri=self._uri_public,
            topics=[f'trade.{symbol}' for symbol in symbols],
            callback=callback
        )
        position_websocket = threading.Thread(target=asyncio.run, args=(connect_promise,))
        position_websocket.start()
        self._websockets.append(position_websocket)

    def _authentication_string(self):
        expires = int((time.time() + 60)) * 1000
        signature = str(
            hmac.new(bytes(self._api_secret, 'utf-8'), bytes(f'GET/realtime{expires}', 'utf-8'),
                     digestmod='sha256').hexdigest())
        return json.dumps({'op': 'auth', 'args': [self._api_key, expires, signature]})

    async def _websocket_connect(self, uri, topics, callback, authenticate=False):
        async with websockets.connect(uri) as ws:
            if authenticate:
                await ws.send(self._authentication_string())
            topics_string = ', '.join(f'"{topic}"' for topic in topics)
            await ws.send('{"op": "subscribe", "args": [' + topics_string + ']}')
            while True:
                rcv = await ws.recv()
                callback(json.loads(rcv))
"""


class MinutePriceBuffer:
    def __init__(self, symbol_count: int):
        self.symbol_count = symbol_count

        # Only used while downloading historical data
        self.last_idx = 0
        self.history_prices = {}

        # Price buffer
        self.buffer_prices = collections.deque(maxlen=7 * 24 * 60)

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


class BinanceWebsocket:
    def __init__(self, api_key: str, api_secret: str):
        self._api_key = api_key
        self._api_secret = api_secret
        self._uri = "wss://stream.binance.com:9443/ws"
        self._websockets = []

    def position(self, callback):
        connect_promise = self._websocket_connect(
            uri=self._uri,
            topics=["position"],
            callback=callback,
            authenticate=True
        )
        position_websocket = threading.Thread(target=asyncio.run, args=(connect_promise,))
        position_websocket.start()
        self._websockets.append(position_websocket)

    def trades(self, symbols, callback):
        connect_promise = self._websocket_connect(
            uri=self._uri,
            topics=[f'{symbol.lower()}@aggTrade' for symbol in symbols],
            callback=callback
        )
        position_websocket = threading.Thread(target=asyncio.run, args=(connect_promise,))
        position_websocket.start()
        self._websockets.append(position_websocket)

    def _authentication_string(self):
        expires = int((time.time() + 60)) * 1000
        signature = str(
            hmac.new(bytes(self._api_secret, 'utf-8'), bytes(f'GET/realtime{expires}', 'utf-8'),
                     digestmod='sha256').hexdigest())
        return json.dumps({'op': 'auth', 'args': [self._api_key, expires, signature]})

    async def _websocket_connect(self, uri, topics, callback, authenticate=False):
        while True:
            async with websockets.connect(uri) as ws:
                try:
                    if authenticate:
                        await ws.send(self._authentication_string())
                    topics_string = ','.join(f'"{topic}"' for topic in topics)
                    msg = '{"method": "SUBSCRIBE", "params": [' + topics_string + '], "id": 1}'
                    await ws.send(msg)
                    rcv = await ws.recv()

                    while True:
                        rcv = await ws.recv()
                        callback(json.loads(rcv))
                except websockets.ConnectionClosed:
                    time.sleep(1)


class MinutePriceGetter:
    def __init__(self, minute_price_buffer: MinutePriceBuffer, start_time: datetime, symbols: list):
        self._minute_price_buffer = minute_price_buffer
        self._symbols = symbols
        #self.current_prices = {}
        self._mark_prices = {}

        history_end_time = datetime.now(timezone.utc)
        self._next_timestamp = history_end_time + timedelta(minutes=1)

        with open('credentials.json') as f:
            credentials = json.load(f)

        self._client = Client(
            api_key=credentials['binance']['api_key'],
            api_secret=credentials['binance']['api_secret']
        )

        self._check_symbols()

        self._binance_websocket = BinanceWebsocket(
            api_key=credentials['binance']['api_key'],
            api_secret=credentials['binance']['api_key']
        )
        self._binance_websocket.trades(symbols=self._symbols, callback=self._trades_callback)

        self._download_history(start_time=start_time, end_time=history_end_time)

    def _trades_callback(self, trade):
        #print("Trade callback", trade)
        symbol = trade['s']
        if symbol not in self._symbols:
            return

        self._mark_prices[symbol] = float(trade['p'])

        current_timestamp = datetime.now(timezone.utc)
        if current_timestamp > self._next_timestamp and len(self._mark_prices) == len(self._symbols):
            print("Appending", self._next_timestamp, self._mark_prices)
            self._next_timestamp = self._next_timestamp + timedelta(minutes=1)
            self._minute_price_buffer.append(self._mark_prices.copy())

    def _check_symbols(self):
        all_symbols = set()
        exchange_info = self._client.get_exchange_info()
        for symbol in exchange_info['symbols']:
            all_symbols.add(symbol['symbol'])

        for symbol in self._symbols:
            if symbol not in all_symbols:
                print(f"MinutePriceGetter, Error {symbol} is not available!")
                quit()

    def _download_history(self, start_time: datetime, end_time: datetime):
        #limiter = pyrate_limiter.Limiter(pyrate_limiter.RequestRate(1, pyrate_limiter.Duration.SECOND))
        #@limiter.ratelimit("download_print_status", delay=False)
        #def print_status(timestamp):
        #    print(f"{datetime.now()} {timestamp}")

        for symbol in self._symbols:
            start_timestamp = int(datetime.timestamp(start_time) * 1000)
            end_timestamp = int(datetime.timestamp(end_time) * 1000)

            for kline in self._client.get_historical_klines_generator(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_1MINUTE,
                    start_str=start_timestamp,
                    end_str=end_timestamp
            ):
                close_price = float(kline[4])
                self._minute_price_buffer.history_append(symbol, close_price)
            print("Downloaded", symbol)

        self._minute_price_buffer.history_finish()


def server():
    symbols = [
        'ADAUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'BTTUSDT', 'CHZUSDT', 'DOGEUSDT', 'EOSUSDT', 'ETCUSDT', 'ETHUSDT',
        'LINKUSDT', 'LTCUSDT', 'MATICUSDT', 'NEOUSDT', 'THETAUSDT', 'TRXUSDT', 'VETUSDT', 'XLMUSDT', 'XRPUSDT'
    ]

    minute_price_buffer = MinutePriceBuffer(symbol_count=len(symbols))

    history_start_time = datetime.now(timezone.utc) - timedelta(minutes=7 * 24 * 60)

    print("Downloading history")

    _minute_price_getter = MinutePriceGetter(minute_price_buffer=minute_price_buffer, start_time=history_start_time, symbols=symbols)

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
            last_idx, prices = minute_price_buffer.get_all()
            send_data = {
                'last_idx': last_idx,
                'prices': prices
            }

        elif command == 'get_since':
            last_idx = payload
            last_idx, prices = minute_price_buffer.get_since(last_idx)
            send_data = {
                'last_idx': last_idx,
                'prices': prices
            }

        socket.send_pyobj(send_data)


if __name__ == "__main__":
    server()
