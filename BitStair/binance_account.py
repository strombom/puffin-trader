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
        self._mark_prices = {}
        self._client = binance_client
        self._total_equity = 0.0

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
        total_equity = 0.0
        for symbol in account_info['balances']:
            if symbol['asset'] == 'USDT':
                total_equity += float(symbol['free'])
            else:
                trade_pair = symbol['asset'] + 'USDT'
                if trade_pair in self._mark_prices:
                    self._balances[trade_pair] = float(symbol['free'])
                    total_equity += self._balances[trade_pair] * self._mark_prices[trade_pair]
        self._total_equity = total_equity
        print(f"Account balance: ", end='')
        for trade_pair in self._balances:
            if self._balances[trade_pair] > 0:
                print(f"{trade_pair}: {self._balances[trade_pair]}  ", end='')
        print(f"Account equity: {self._total_equity} USDT")

    def calculate_leverage(self):
        self._update_account_status()
        mark_price = (self.mark_price_ask + self.mark_price_bid) / 2
        usdt = self.assets['USDT']['wallet'] - self.assets['USDT']['debt']
        btc = (self.assets['BTC']['wallet'] - self.assets['BTC']['debt']) * mark_price
        leverage = btc / (usdt + btc)
        return leverage

    def get_balance(self, trade_pair):
        return self._balances[trade_pair]

    def get_total_equity_usdt(self):
        return 300.0

    def get_mark_price(self, trade_pair):
        return self._mark_prices[trade_pair]

    def market_buy(self, trade_pair, volume):
        print(f"Buy {volume} {trade_pair}")

    def market_sell(self, trade_pair, volume):
        order = self._client.order_market_sell(
            symbol=trade_pair,
            quantity=volume
        )
        if order['status'] != 'FILLED':
            print(f"Market sell  {volume} {trade_pair} FAILED! {order}")
        else:
            print(f"Market sold {volume} {trade_pair} OK")

    def order(self, leverage):
        borrow_extra = 1.01
        min_order_size_usdt = 10

        if leverage == 0:
            return

        # print(f"Order leverage({leverage})")
        if leverage > 0:
            mark_price = self.mark_price_ask
        else:
            mark_price = self.mark_price_bid

        # self.update_account_status()

        equity = self.assets['USDT']['wallet'] - self.assets['USDT']['debt']
        equity += (self.assets['BTC']['wallet'] - self.assets['BTC']['debt']) * mark_price

        target_btc = leverage * equity / mark_price

        order_size_btc = target_btc + self.assets['BTC']['debt'] - self.assets['BTC']['wallet']

        print(f"Order size BTC: leverage({leverage}) target_btc({target_btc}) order_size_btc({order_size_btc})")

        # Trade BTC<->USDT
        if leverage > 0 and order_size_btc * mark_price > min_order_size_usdt:
            min_lot_size = 0.000001
            order_size_btc = math.floor(order_size_btc / min_lot_size) * min_lot_size
            order = self._client.create_margin_order(
                symbol='BTCUSDT',
                side=binance.enums.SIDE_BUY,
                type=binance.enums.ORDER_TYPE_MARKET,
                quantity=str(order_size_btc),
                newOrderRespType=binance.enums.ORDER_RESP_TYPE_RESULT,
                sideEffectType='MARGIN_BUY')
            print(f"Order BTC ({order_size_btc})")
            print("Order BTC result", order)

        elif leverage < 0 and -order_size_btc * mark_price > min_order_size_usdt:
            min_lot_size = 0.000001
            order_size_btc = math.floor(order_size_btc / min_lot_size) * min_lot_size
            order = self._client.create_margin_order(
                symbol='BTCUSDT',
                side=binance.enums.SIDE_SELL,
                type=binance.enums.ORDER_TYPE_MARKET,
                quantity=str(-order_size_btc),
                newOrderRespType=binance.enums.ORDER_RESP_TYPE_RESULT,
                sideEffectType='MARGIN_BUY')
            print(f"Order BTC ({order_size_btc})")
            print("Order BTC result", order)

        # Repay debt
        for retry in range(3):
            self.update_account_status()
            try:
                if self.assets['BTC']['wallet'] > 0 and self.assets['BTC']['debt'] > 0:
                    repay_btc = min(self.assets['BTC']['wallet'], self.assets['BTC']['debt'])
                    details = self._client.repay_margin_loan(asset='BTC', amount=str(repay_btc))
                    print(f"Repay BTC ({repay_btc})")
                    print("Repay BTC result", details)
                break
            except binance.exceptions.BinanceAPIException:
                traceback.print_exc()

        for retry in range(3):
            self.update_account_status()
            try:
                if self.assets['USDT']['wallet'] > 0 and self.assets['USDT']['debt'] > 0:
                    repay_usdt = min(self.assets['USDT']['wallet'], self.assets['USDT']['debt'])
                    details = self._client.repay_margin_loan(asset='USDT', amount=str(repay_usdt))
                    print(f"Repay USDT ({repay_usdt})")
                    print("Repay USDT result", details)
                break
            except binance.exceptions.BinanceAPIException:
                traceback.print_exc()

        sleep(0.3)
        self.update_account_status()
