import json
import math
import traceback
from time import sleep

import binance.enums
import binance.exceptions
from binance.websockets import BinanceSocketManager


class BinanceAccount:
    assets = {}
    mark_price_ask = 0
    mark_price_bid = 0

    def __init__(self, binance_client):
        self.client = binance_client

        # exchange_info = self.client.get_exchange_info()
        # for symbol in exchange_info['symbols']:
        #     if symbol['symbol'] == 'BTCUSDT':
        #         print("Exchange info", symbol)

        def process_margin_message(data):
            print("margin", data)

        self.margin_socket_manager = BinanceSocketManager(self.client)
        self.margin_socket_manager.start_margin_socket(process_margin_message)
        self.margin_socket_manager.start()

        def process_trade_message(data):
            if data['e'] == 'trade' and data['s'] == 'BTCUSDT':
                price = float(data['p'])
                if data['m']:
                    self.mark_price_bid = price
                else:
                    self.mark_price_ask = price

        self.trade_socket_manager = BinanceSocketManager(self.client)
        self.trade_socket_manager.start_trade_socket('BTCUSDT', process_trade_message)
        self.trade_socket_manager.start()

        from time import sleep
        while self.mark_price_ask == 0 or self.mark_price_bid == 0:
            sleep(1)

        # transaction = self.client.create_margin_loan(asset='BTC', amount='0.00001')
        # print(f'transaction {transaction}')
        # transaction {'tranId': 42554065113, 'clientTag': ''}

        # transaction = self.client.create_margin_loan(asset='USDT', amount='1.00000')
        # print(f'transaction {transaction}')
        # transaction {'tranId': 42555634933, 'clientTag': ''}

        # margin_usdt = self.client.get_margin_asset(asset='USDT')
        # margin_loan_details = self.client.get_margin_loan_details(asset='BTC', startTime=0)
        # print(f'margin_loan_details {margin_loan_details}')
        # margin_usdt {'assetName': 'USDT', 'assetFullName': 'Tether', 'isBorrowable': True, 'isMortgageable': True, 'userMinBorrow': '0', 'userMinRepay': '0'}

        self.update_account_status()

        return

        self.order(2.0)
        self.order(1.5)
        self.order(1.0)
        self.order(0.5)
        self.order(0.0)
        self.order(-0.5)
        self.order(-1.0)
        self.order(-1.5)
        self.order(-2.0)

        quit()

        """
        asset_balance_btc = self.client.get_asset_balance(asset='BTC')
        print(f'asset_balance_btc {asset_balance_btc}')

        asset_balance_usdt = self.client.get_asset_balance(asset='USDT')
        print(f'asset_balance_usdt {asset_balance_usdt}')

        account_status = self.client.get_account_status()
        print(f'account_status {account_status}')
        """

        """
        quit()
        order = client.create_margin_order(
            symbol='BNBBTC',
            side=SIDE_BUY,
            type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=100,
            price='0.00001')
        """

    def update_account_status(self):
        account_info = self.client.get_margin_account()
        for asset in account_info['userAssets']:
            name = asset['asset']
            self.assets[name] = {'wallet': float(asset['free']),
                                 'debt': float(asset['borrowed'])}
        print(f"Account BTC wallet={self.assets['BTC']['wallet']}, debt={self.assets['BTC']['debt']}")
        print(f"Account USDT wallet={self.assets['USDT']['wallet']}, debt={self.assets['USDT']['debt']}")

    def calculate_leverage(self):
        self.update_account_status()
        mark_price = (self.mark_price_ask + self.mark_price_bid) / 2
        usdt = self.assets['USDT']['wallet'] - self.assets['USDT']['debt']
        btc = (self.assets['BTC']['wallet'] - self.assets['BTC']['debt']) * mark_price
        leverage = btc / (usdt + btc)
        return leverage

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
            order = self.client.create_margin_order(
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
            order = self.client.create_margin_order(
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
                    details = self.client.repay_margin_loan(asset='BTC', amount=str(repay_btc))
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
                    details = self.client.repay_margin_loan(asset='USDT', amount=str(repay_usdt))
                    print(f"Repay USDT ({repay_usdt})")
                    print("Repay USDT result", details)
                break
            except binance.exceptions.BinanceAPIException:
                traceback.print_exc()

        sleep(0.3)
        self.update_account_status()
