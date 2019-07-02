
import time
import json

from token_db import TokenDB

from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient
from bitmex_authenticator import APIKeyAuthenticator
from exchange_keys import bitmex_keys


class Bitmex:
    def __init__(self):
        config = {
            'use_models': False,
            'validate_responses' : False,
            'also_return_response': True
        }

        if False:
            host = 'https://testnet.bitmex.com'
        else:
            host = 'https://www.bitmex.com'

        spec_uri = host + '/api/explorer/swagger.json'

        request_client = RequestsClient()
        request_client.authenticator = APIKeyAuthenticator(host, bitmex_keys['api_key'], bitmex_keys['api_secret'])

        self.client = SwaggerClient.from_url(spec_uri, config = config, http_client = request_client)

    def get_instrument(self):
        result = self.client.Instrument.Instrument_get(filter=json.dumps({'symbol': 'XBTUSD'})).result()
        return result

    def get_first_tick_aggregate_id(self, symbol, base_symbol, timestamp):
        binance_symbol = symbol + base_symbol
        trades = self.client.get_aggregate_trades(symbol = binance_symbol,
                                                  startTime = timestamp - 1000 * 60 * 60,
                                                  endTime = timestamp,
                                                  limit = 1)
        tick_aggregate_id = trades[0]['a'] + 1
        return tick_aggregate_id

    def get_ticks(self, symbol, base_symbol, aggregate_id_start = 0, limit = None):
        binance_symbol = symbol + base_symbol
        # https://python-binance.readthedocs.io/en/latest/binance.html?highlight=aggregate_trades#binance.client.Client.get_aggregate_trades
        trades = self.client.get_aggregate_trades(symbol = binance_symbol,
                                                  fromId = aggregate_id_start,
                                                  limit = limit)
        ticks = []
        for trade in trades:
            tick = {'timestamp': trade['T'],
                    'tick_aggregate_id': trade['a'],
                    'price': trade['p'],
                    'volume': trade['q']}
            ticks.append(tick)
        return ticks







bitmex = Bitmex()
quote = bitmex.get_quote()
print(quote)


quit()

binance = Binance()

token_db = TokenDB('token_btc_usdt.db')

symbol = "BTC"
base_symbol = 'USDT'

while True:
    timeout = time.time() + 0.2

    token_state = token_db.get_token_state()
    if token_state is None:
        #try:
        first_timestamp = 1536148800000
        tick_aggregate_id = binance.get_first_tick_aggregate_id(symbol, base_symbol, first_timestamp)
        token_db.init_token(tick_aggregate_id)
        time.sleep(0.5)
        print("continue")
        continue
        #except:
        #    break

        print("ts", token_state)
    ticks_next_tick_aggregate_id = token_state['ticks_next_tick_aggregate_id']
    for retry in range(3):
        try:
            ticks = binance.get_ticks(symbol, base_symbol, ticks_next_tick_aggregate_id, limit = 1000)
            if len(ticks) == 0:
                break
            print("Appending data (", symbol, base_symbol, ")", ticks_next_tick_aggregate_id, len(ticks))
            token_db.append_ticks(ticks)
            break
        except:
            pass
    
    # Update volume data
    while True:
        token_state = token_db.get_token_state()

        ticks = token_db.get_ticks(tick_aggregate_id = token_state['volume_next_tick_aggregate_id'], limit = 10000)
        if not ticks:
            break
        
        volume_data = {'price_low': [],
                       'price_high': [],
                       'timestamp': []}

        last_tick_id = token_state['volume_next_tick_aggregate_id']
        remaining_volume = token_state['volume_remaining_volume']
        price_high = token_state['volume_remaining_price_high']
        price_low = token_state['volume_remaining_price_low']
        
        if remaining_volume == 0:
            price_high = 0
            price_low = 10**10
            
        for idx, tick_id in enumerate(ticks['tick_aggregate_id']):
            timestamp = ticks['timestamp'][idx]
            price = ticks['price'][idx]
            volume = ticks['volume'][idx]
            
            if price < price_low:
                price_low = price
            if price > price_high:
                price_high = price

            while remaining_volume + volume >= TokenDB.volume_stepsize:
                volume_data['price_high'].append(price_high)
                volume_data['price_low'].append(price_low)
                volume_data['timestamp'].append(timestamp)

                volume -= TokenDB.volume_stepsize - remaining_volume

                remaining_volume = 0
                remaining_price_high = price_high
                remaining_price_low = price_low

                if volume == 0:
                    price_high = 0
                    price_low = 10**10
                else:
                    price_high = price
                    price_low = price
                    remaining_price_high = price
                    remaining_price_low = price

            last_tick_id = tick_id
            remaining_volume += volume

        if len(volume_data['timestamp']):
            token_db.append_trade_data_volume(volume_data, last_tick_id, remaining_volume, remaining_price_high, remaining_price_low)
            volume_data['price_high'] = []
            volume_data['price_low'] = []
            volume_data['timestamp'] = []
        else:
            break
            
    while time.time() < timeout:
        time.sleep(0.05)
