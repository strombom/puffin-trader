import math
import asyncio
import threading

from binance import AsyncClient
from binance import BinanceSocketManager


class BinanceAccount:
    def __init__(self, api_key, api_secret, symbols):
        self._api_key = api_key
        self._api_secret = api_secret
        self._balances = {}
        self._balance_usdt = 0.0
        self._mark_prices = {}
        self._tick_sizes = {}
        self._min_lot_sizes = {}
        self._total_equity = 0.0
        self._kline_threads = {}

        self.event_loop = asyncio.get_event_loop()
        self.event_loop.run_until_complete(self.__async__init(symbols))
        self._update_account_status()

    def kline_thread(self, symbol: str, loop):
        async def kline_task():
            ts = self._socket_manager.kline_socket(symbol, interval=self._client.KLINE_INTERVAL_1MINUTE)
            async with ts as ts_c:
                while True:
                    data = await ts_c.recv()
                    if data['e'] == 'kline':
                        self._mark_prices[data['s']] = float(data['k']['c'])
                        # print("set price", symbol, float(data['k']['c']))

        loop.run_until_complete(kline_task())

    async def __async__init(self, symbols: str):
        self._client = await AsyncClient.create(api_key=self._api_key, api_secret=self._api_secret)
        self._socket_manager = BinanceSocketManager(self._client)

        info = await self._client.get_exchange_info()
        for symbol in info['symbols']:
            for symbol_filter in symbol['filters']:
                if symbol_filter['filterType'] == "LOT_SIZE":
                    # TODO: Potential bug if tick size doesn't end in 1
                    self._tick_sizes[symbol['symbol']] = int(math.log10(float(symbol_filter['stepSize'])))
                    self._min_lot_sizes[symbol['symbol']] = float(symbol_filter['minQty'])
                    break

        print("symbols", symbols)

        tickers = await self._client.get_all_tickers()
        for ticker in tickers:
            if ticker['symbol'] in symbols:
                self._mark_prices[ticker['symbol']] = float(ticker['price'])

        for symbol in symbols:
            self._kline_threads[symbol] = threading.Thread(target=self.kline_thread, args=(symbol, asyncio.new_event_loop()))
            self._kline_threads[symbol].start()

    def get_portfolio(self):
        self._update_account_status()
        portfolio = set()
        for trade_pair in self._balances:
            if self._balances[trade_pair] > 0:
                portfolio.add(trade_pair)
        return portfolio

    def _update_account_status(self):
        account_info = self.event_loop.run_until_complete(self._client.get_account())
        for symbol in account_info['balances']:
            if symbol['asset'] == 'USDT':
                self._balance_usdt = float(symbol['free'])
            else:
                trade_pair = symbol['asset'] + 'USDT'
                if trade_pair in self._mark_prices:
                    balance = float(symbol['free'])
                    if balance < self._min_lot_sizes[trade_pair]:
                        balance = 0.0
                    self._balances[trade_pair] = balance

        self._total_equity = self.get_total_equity_usdt()
        print(f"Account balance: ", end='')
        for trade_pair in self._balances:
            if self._balances[trade_pair] > 0:
                print(f"{trade_pair}: {self._balances[trade_pair]}  ", end='')
        print(f"Account equity: {self._total_equity} USDT")

    def get_balance(self, trade_pair):
        return self._balances[trade_pair]

    def get_balance_usdt(self):
        return self._balance_usdt

    def get_total_equity_usdt(self):
        total_equity = self._balance_usdt
        for trade_pair in self._balances:
            total_equity += self._balances[trade_pair] * self._mark_prices[trade_pair]
        return total_equity

    def get_mark_price(self, trade_pair):
        return self._mark_prices[trade_pair]

    def market_buy(self, trade_pair, volume):
        factor = 10 ** -self._tick_sizes[trade_pair]
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
        factor = 10 ** -self._tick_sizes[trade_pair]
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
