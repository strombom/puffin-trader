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


def bybit_websocket_authentication_string():
    key_id = '7KbQsjaCpUELgkBJXN'
    key_secret = 'SNM2E0u8vvQdlWtnVxqlwklMVSTQqT8Ji13b'
    expires = int((time.time() + 60)) * 1000
    signature = str(
        hmac.new(bytes(key_secret, 'utf-8'), bytes(f'GET/realtime{expires}', 'utf-8'), digestmod='sha256').hexdigest())
    return json.dumps({'op': 'auth', 'args': [key_id, expires, signature]})


class BybitAccount:
    def __init__(self, api_key, api_secret, symbols):
        self._api_key = api_key
        self._api_secret = api_secret
        self._symbols = symbols
        #self._debt = {}
        self._balance = {}
        self._mark_price = {}
        self._tick_size = {}
        self._min_lot_size = {}
        self._min_notional = {}
        self._balance_lock = threading.Lock()
        self._bybit_rest = BybitRest(api_key=api_key, api_secret=api_secret)
        self._bybit_websocket = BybitWebsocket(api_key=api_key, api_secret=api_secret)

        self._filter_symbols()

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

    def _start_websocket(self):
        uri_private = "wss://stream.bybit.com/realtime_private"
        uri_public = "wss://stream.bybit.com/realtime_public"
        self.position_websocket = threading.Thread(target=asyncio.run, args=(self.websocket(uri_private, "position"),))
        self.position_websocket.start()

    async def websocket(self, uri, topic):
        async with websockets.connect(uri) as ws:
            await ws.send(bybit_websocket_authentication_string())
            await ws.send('{"op": "subscribe", "args": ["' + topic + '"]}')
            while True:
                rcv = await ws.recv()
                print(f"< {rcv}")


class BybitRest:
    def __init__(self, api_key: str, api_secret: str):
        self._api_key = api_key
        self._api_secret = api_secret

    def get_symbols(self):
        return self.get(endpoint="/v2/public/symbols", params={})

    def auth_signature(self, params: dict):
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

    def get(self, endpoint: str, params: dict):
        params['api_key'] = self._api_key
        params['timestamp'] = str(int(datetime.now().timestamp() * 1000))
        signature = self.auth_signature(params)

        url = f'https://api.bybit.com{endpoint}?'
        for key in sorted(params.keys()):
            value = params[key]
            url += f'{key}={value}&'
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
