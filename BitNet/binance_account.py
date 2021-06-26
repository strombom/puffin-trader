import math
import asyncio
import threading
from datetime import datetime, timezone, timedelta

import binance.enums
from binance import ThreadedWebsocketManager, Client
from binance.exceptions import BinanceAPIException


class BinanceAccount:
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

        self._client = Client(api_key=self._api_key, api_secret=self._api_secret)

        info = self._client.get_exchange_info()
        for symbol in info['symbols']:
            if symbol['symbol'] not in symbols:
                continue
            for symbol_filter in symbol['filters']:
                if symbol_filter['filterType'] == "LOT_SIZE":
                    # TODO: Potential bug if tick size doesn't end in 1
                    self._tick_size[symbol['symbol']] = int(math.log10(float(symbol_filter['stepSize'])))
                    self._min_lot_size[symbol['symbol']] = float(symbol_filter['minQty'])
                if symbol_filter['filterType'] == "MIN_NOTIONAL":
                    self._min_notional[symbol['symbol']] = float(symbol_filter['minNotional'])

        self.twm = ThreadedWebsocketManager(
            api_key=self._api_key,
            api_secret=self._api_secret
        )
        self.twm.start()

        # Get margin account balances
        with self._balance_lock:
            account_info = self._client.get_margin_account()
            for asset in account_info['userAssets']:
                if asset['asset'] + 'USDT' in symbols or asset['asset'] == 'USDT':
                    # print(f"Update balance {asset['free']} {asset['asset']}")
                    self._balance[asset['asset']] = float(asset['free'])
                    #self._debt[asset['asset']] = asset['borrowed']

        # Get all prices
        prices = self._client.get_all_tickers()
        for price in prices:
            if price['symbol'] in symbols:
                self._mark_price[price['symbol']] = float(price['price'])

        streams = [f"{symbol.lower()}@trade" for symbol in self._symbols]
        self.twm.start_multiplex_socket(callback=self._process_tick_message, streams=streams)
        self.twm.start_margin_socket(callback=self._process_margin_message)

    def _process_margin_message(self, data):
        with self._balance_lock:
            if data['e'] == 'outboundAccountPosition':
                for position in data['B']:
                    asset = position['a']
                    quantity = float(position['f'])
                    if asset not in self._balance or quantity != self._balance[asset]:
                        self._balance[asset] = quantity
                        # print(f"Update balance {quantity} {asset}")

    def _process_tick_message(self, data):
        symbol = data['data']['s']
        last_price = float(data['data']['p'])
        #if symbol in self._mark_prices and self._mark_prices[symbol] != last_price:
        #    print(f"Process tick message {symbol} {last_price}")
        self._mark_price[symbol] = last_price

    """
    def get_portfolio(self):
        portfolio = set()
        for trade_pair in self._balance:
            if self._balance[trade_pair] > 0:
                portfolio.add(trade_pair)
        return portfolio
    """

    def get_balance(self, asset):
        return self._balance[asset]

    #def get_balance_usdt(self):
    #    return self._balance['USDT']

    def get_total_equity_usdt(self):
        total_equity = self._balance['USDT']
        for asset in self._balance:
            if asset == 'USDT':
                continue
            total_equity += self._balance[asset] * self._mark_price[asset + 'USDT']
        return total_equity

    def get_mark_price(self, symbol):
        return self._mark_price[symbol]

    def sell_all(self):
        for asset in self._balance:
            if asset != 'USDT' and self._balance[asset] > 0:
                self.market_sell(symbol=asset + 'USDT', volume=self._balance[asset])

    def market_buy(self, symbol, volume):
        factor = 10 ** -self._tick_size[symbol]
        quantity = math.floor(volume * factor) / factor
        if quantity == 0:
            print(f"Market buy {volume} {symbol} FAILED! Volume too low.")
            return {'quantity': 0, 'price': 0}
        elif quantity < self._min_lot_size[symbol]:
            print(f"Market buy {volume} {symbol} FAILED! Volume below min lot size: {self._min_lot_size[symbol]}.")
            return {'quantity': 0, 'price': 0}
        elif quantity * self._mark_price[symbol] < self._min_notional[symbol]:
            print(f"Market buy {volume} {symbol} FAILED! Price * quantity below min notional: {self._min_notional[symbol]}.")
            return {'quantity': 0, 'price': 0}

        for retry in range(3):
            try:
                order = self._client.create_margin_order(
                    symbol=symbol,
                    side=binance.enums.SIDE_BUY,
                    type=binance.enums.ORDER_TYPE_MARKET,
                    quantity=quantity
                )

                if order['status'] != 'FILLED':
                    print(f"Market buy  {quantity} {symbol} FAILED! {order}")
                    return {'quantity': 0, 'price': 0}
                else:
                    print(f"Market buy {quantity} {symbol} OK")
                    return {'quantity': float(order['executedQty']), 'price': float(order['price'])}

            except BinanceAPIException as e:
                print(f"Market buy  {quantity} {symbol} error: {e}")

        print(f"Market buy  {quantity} {symbol} FAILED!")
        return {'quantity': 0, 'price': 0}

    def market_sell(self, symbol, volume):
        factor = 10 ** -self._tick_size[symbol]
        quantity = math.floor(volume * factor) / factor
        if quantity == 0:
            print(f"Market sell {volume} {symbol} FAILED! Volume too low.")
            return {'quantity': 0, 'price': 0}

        for retry in range(3):
            try:
                order = self._client.create_margin_order(
                    symbol=symbol,
                    side=binance.enums.SIDE_SELL,
                    type=binance.enums.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                if order['status'] != 'FILLED':
                    print(f"Market sell {quantity} {symbol} FAILED! {order}")
                    return {'quantity': 0, 'price': 0}
                else:
                    print(f"Market sell {quantity} {symbol} OK")
                    return {'quantity': float(order['executedQty']), 'price': float(order['price'])}

            except BinanceAPIException as e:
                print(f"Market sell  {quantity} {symbol} error: {e}")

        print(f"Market sell  {quantity} {symbol} FAILED!")
        return {'quantity': 0, 'price': 0}
