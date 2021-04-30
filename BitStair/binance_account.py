import json
import math
import traceback
from time import sleep

import binance.enums
import binance.exceptions
from binance.websockets import BinanceSocketManager


class BinanceAccount:
    def __init__(self, binance_client, trade_pairs):
        self._balances = {}
        self._balance_usdt = 0.0
        self._mark_prices = {}
        self._tick_sizes = {}
        self._client = binance_client
        self._total_equity = 0.0

        info = self._client.get_exchange_info()
        for symbol in info['symbols']:
            for symbol_filter in symbol['filters']:
                if symbol_filter['filterType'] == "PRICE_FILTER":
                    self._tick_sizes[symbol['symbol']] = int(math.log10(float(symbol_filter['tickSize'])))
                    break

        tickers = self._client.get_all_tickers()
        for ticker in tickers:
            if ticker['symbol'] in trade_pairs:
                self._mark_prices[ticker['symbol']] = float(ticker['price'])

        def process_kline_message(data):
            if data['e'] == 'kline':
                self._mark_prices[data['s']] = float(data['k']['c'])

        self._kline_threads = {}
        for trade_pair in trade_pairs:
            kline_socket_manager = BinanceSocketManager(self._client)
            kline_socket_manager.start_kline_socket(symbol=trade_pair, callback=process_kline_message, interval='1m')
            kline_socket_manager.start()
            self._kline_threads[trade_pair] = kline_socket_manager

        self._update_account_status()

    def get_portfolio(self):
        portfolio = set()
        for trade_pair in self._balances:
            if self._balances[trade_pair] > 0:
                portfolio.add(trade_pair)
        return portfolio

    def _update_account_status(self):
        account_info = self._client.get_account()
        for symbol in account_info['balances']:
            if symbol['asset'] == 'USDT':
                self._balance_usdt = float(symbol['free'])
            else:
                trade_pair = symbol['asset'] + 'USDT'
                if trade_pair in self._mark_prices:
                    self._balances[trade_pair] = float(symbol['free'])
        self._total_equity = self.get_total_equity_usdt()
        print(f"Account balance: ", end='')
        for trade_pair in self._balances:
            if self._balances[trade_pair] > 0:
                print(f"{trade_pair}: {self._balances[trade_pair]}  ", end='')
        print(f"Account equity: {self._total_equity} USDT")

    def get_balance(self, trade_pair):
        return self._balances[trade_pair]

    def get_total_equity_usdt(self):
        total_equity = self._balance_usdt
        for trade_pair in self._balances:
            total_equity += self._balances[trade_pair] * self._mark_prices[trade_pair]
        return total_equity

    def get_mark_price(self, trade_pair):
        return self._mark_prices[trade_pair]

    def market_buy(self, trade_pair, volume):
        quantity = round(volume, -self._tick_sizes[trade_pair])
        order = self._client.order_market_buy(
            symbol=trade_pair,
            quantity=quantity
        )
        if order['status'] != 'FILLED':
            print(f"Market buy  {quantity} {trade_pair} FAILED! {order}")
        else:
            print(f"Market buy {quantity} {trade_pair} OK")

    def market_sell(self, trade_pair, volume):
        quantity = round(volume, -self._tick_sizes[trade_pair])
        order = self._client.order_market_sell(
            symbol=trade_pair,
            quantity=quantity
        )
        if order['status'] != 'FILLED':
            print(f"Market sell  {quantity} {trade_pair} FAILED! {order}")
        else:
            print(f"Market sell {quantity} {trade_pair} OK")
