import asyncio
import hashlib
import hmac
import json
import time
import threading
from datetime import datetime

import requests
import urllib3
import websockets


class BybitAccount:
    def __init__(self, api_key, api_secret, symbols):
        self._api_key = api_key
        self._api_secret = api_secret
        self._symbols = symbols
        self._positions = {}
        self._mark_price = {}
        #self._tick_size = {}
        #self._min_lot_size = {}
        #self._min_notional = {}
        self._balance_lock = threading.Lock()
        self._equity = {}

        self._bybit_rest = BybitRest(api_key=api_key, api_secret=api_secret)
        self._filter_symbols()
        self._update_mark_prices()

        self._bybit_websocket = BybitWebsocket(api_key=api_key, api_secret=api_secret)
        self._bybit_websocket.position(callback=self._position_callback)
        self._bybit_websocket.trades(symbols=self._symbols, callback=self._trades_callback)

        self.update_balance()

    def has_symbol(self, symbol):
        return symbol in self._symbols

    def get_total_equity_usdt(self):
        total_equity_usdt = 0
        for symbol in self._equity:
            if symbol == 'USDT':
                mark_price = 1
            else:
                mark_price = self._mark_price[symbol + 'USDT']
            total_equity_usdt += self._equity[symbol] * mark_price
        return total_equity_usdt

    def get_balance(self, symbol):
        if symbol == 'USDT':
            positions_value_usdt = 0
            for position_symbol in self._positions:
                positions_value_usdt += abs(self._positions[position_symbol]) * self._mark_price[position_symbol]
            return self.get_total_equity_usdt() - positions_value_usdt

        self._positions[symbol] * self._mark_price[symbol]

    def get_mark_price(self, symbol):
        return self._mark_price[symbol]

    def market_buy(symbol, volume):
        print(1)

    def update_balance(self):
        with self._balance_lock:
            wallet_balance = self._bybit_rest.get_wallet_balance()
            positions = self._bybit_rest.get_positions()

            for symbol in wallet_balance['result']:
                self._equity[symbol] = wallet_balance['result'][symbol]['equity']

            self._positions.clear()
            for position in positions['result']:
                symbol = position['data']['symbol']
                if position['is_valid'] and symbol in self._symbols:
                    position_size = position['data']['size']
                    if symbol not in self._positions:
                        self._positions[symbol] = 0
                    if position['data']['side'] == 'Buy':
                        self._positions[symbol] += position_size
                    else:
                        self._positions[symbol] -= position_size

    def _update_mark_prices(self):
        self._mark_price = {
            'ADAUSDT': 2.9802, 'BCHUSDT': 717.28, 'BNBUSDT': 490.08, 'BTCUSDT': 50351.81, 'DOGEUSDT': 0.2979,
            'EOSUSDT': 5.673, 'ETCUSDT': 71.093, 'ETHUSDT': 3957.37, 'LINKUSDT': 31.065, 'LTCUSDT': 214.21,
            'MATICUSDT': 1.4774, 'THETAUSDT': 7.296, 'TRXUSDT': 0.10269, 'XLMUSDT': 0.37416, 'XRPUSDT': 1.3048
        }
        #for symbol in self._symbols:
        #    mark_price_kline = self._bybit_rest.get_mark_price_kline(symbol=symbol)
        #    self._mark_price[symbol] = mark_price_kline['close']

    def _position_callback(self, positions):
        if not ('topic' in positions and positions['topic'] == 'position'):
            return

        with self._balance_lock:
            new_positions = {}
            for position in positions['data']:
                symbol = position['symbol']
                if symbol in self._symbols:
                    if symbol not in new_positions:
                        new_positions[symbol] = 0
                    position_size = position['size']
                    if position['side'] == 'Buy':
                        new_positions[symbol] += position_size
                    else:
                        new_positions[symbol] -= position_size

            self._positions.update(new_positions)

    def _trades_callback(self, trades):
        if 'data' not in trades:
            return
        last_trade = trades['data'][-1]
        symbol = last_trade['symbol']
        self._mark_price[symbol] = float(last_trade['price'])

    def _filter_symbols(self):
        self._symbols = ['ADAUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT', 'EOSUSDT', 'ETCUSDT', 'ETHUSDT',
                         'LINKUSDT', 'LTCUSDT', 'MATICUSDT', 'THETAUSDT', 'TRXUSDT', 'XLMUSDT', 'XRPUSDT']
        """
        bybit_symbols = self._bybit_rest.get_symbols()
        new_symbols = []
        for symbol in bybit_symbols['result']:
            if symbol['quote_currency'] == 'USDT':
                if symbol['name'] in self._symbols:
                    new_symbols.append(symbol['name'])
        self._symbols = sorted(new_symbols)
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


class BybitRest:
    def __init__(self, api_key: str, api_secret: str):
        self._api_key = api_key
        self._api_secret = api_secret

    def get_symbols(self):
        symbols = self._get(endpoint="/v2/public/symbols", params={})
        return symbols

    def get_mark_price_kline(self, symbol: str):
        timestamp = str(int(datetime.now().timestamp()) - 120)
        mark_price_kline = self._get(endpoint="/public/linear/mark-price-kline", params={
            'symbol': symbol,
            'interval': '1',
            'from': timestamp
        })
        return mark_price_kline['result'][-1]

    def get_wallet_balance(self):
        wallet_balance = self._get(endpoint='/v2/private/wallet/balance', params={}, authenticate=True)
        return wallet_balance

    def get_positions(self):
        positions = self._get(endpoint='/private/linear/position/list', params={}, authenticate=True)
        return positions

    def _auth_signature(self, params: dict):
        sign = ''
        for key in sorted(params.keys()):
            if isinstance(params[key], bool):
                if params[key]:
                    v = 'true'
                else:
                    v = 'false'
            else:
                v = params[key]
            sign += key + '=' + v + '&'
        sign = sign[:-1]
        return hmac.new(bytearray(self._api_secret.encode()), sign.encode("utf-8"), hashlib.sha256).hexdigest()

    def _get(self, endpoint: str, params: dict, authenticate=False):
        if authenticate:
            params['api_key'] = self._api_key
            params['timestamp'] = str(int(datetime.now().timestamp() * 1000))
            signature = self._auth_signature(params)

        url = f'https://api.bybit.com{endpoint}?'
        for key in sorted(params.keys()):
            value = params[key]
            url += f'{key}={value}&'

        if authenticate:
            url += f'sign={signature}'

        headers = {"Content-Type": "application/json"}
        urllib3.disable_warnings()
        s = requests.session()
        s.keep_alive = False
        response = requests.get(url, headers=headers, verify=False)

        return json.loads(response.text)


if __name__ == '__main__':
    all_symbols = ['ADAUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'BTTUSDT', 'CHZUSDT', 'DOGEUSDT', 'EOSUSDT', 'ETCUSDT',
                   'ETHUSDT', 'LINKUSDT', 'LTCUSDT', 'MATICUSDT', 'NEOUSDT', 'THETAUSDT', 'TRXUSDT', 'VETUSDT',
                   'XLMUSDT', 'XRPUSDT']
    with open('credentials.json') as f:
        credentials = json.load(f)
    bybit_account = BybitAccount(
        api_key=credentials['bybit']['api_key'],
        api_secret=credentials['bybit']['api_secret'],
        symbols=all_symbols
    )
    while True:
        time.sleep(1)

