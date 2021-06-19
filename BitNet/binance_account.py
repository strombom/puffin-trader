import math
import asyncio
import threading
from datetime import datetime, timezone, timedelta

from binance import ThreadedWebsocketManager, Client


#from binance import AsyncClient, BinanceSocketManager


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
                    break

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
                    print(f"Update balance {asset['free']} {asset['asset']}")
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
                    quantity = position['f']
                    if asset not in self._balance or quantity != self._balance[asset]:
                        self._balance[asset] = quantity
                        print(f"Update balance {quantity} {asset}")

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

    def get_balance(self, trade_pair):
        return self._balance[trade_pair]

    #def get_balance_usdt(self):
    #    return self._balance['USDT']

    def get_total_equity_usdt(self):
        total_equity = self._balance['USDT']
        for asset in self._balance:
            if asset == 'USDT':
                continue
            total_equity += self._balance[asset] * self._mark_price[asset + 'USDT']
        return total_equity

    def get_mark_price(self, trade_pair):
        return self._mark_price[trade_pair]

    def market_buy(self, trade_pair, volume):
        factor = 10 ** -self._tick_size[trade_pair]
        quantity = math.floor(volume * factor) / factor
        if quantity == 0:
            return

        order = self.event_loop.run_until_complete(
            self._client.order_market_buy(
                symbol=trade_pair,
                quantity=quantity
            )
        )
        if order['status'] != 'FILLED':
            print(f"Market buy  {quantity} {trade_pair} FAILED! {order}")
        else:
            print(f"Market buy {quantity} {trade_pair} OK")

    def market_sell(self, trade_pair, volume):
        factor = 10 ** -self._tick_size[trade_pair]
        quantity = math.floor(volume * factor) / factor
        if quantity == 0:
            return
        order = self.event_loop.run_until_complete(
            self._client.order_market_sell(
                symbol=trade_pair,
                quantity=quantity
            )
        )
        if order['status'] != 'FILLED':
            print(f"Market sell  {quantity} {trade_pair} FAILED! {order}")
        else:
            print(f"Market sell {quantity} {trade_pair} OK")
