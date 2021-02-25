import json
import binance.enums
from binance.client import Client
from binance.websockets import BinanceSocketManager

class BinanceAccount:
    assets = {}
    mark_price_ask = 0
    mark_price_bid = 0

    def __init__(self):
        with open('binance_account.json') as f:
            account_info = json.load(f)
            api_key = account_info['api_key']
            api_secret = account_info['api_secret']

        self.client = Client(api_key, api_secret)

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

    def order(self, leverage):
        if leverage == 0:
            return

        # print(f"Order leverage({leverage})")
        if leverage > 0:
            mark_price = self.mark_price_ask
        else:
            mark_price = self.mark_price_bid

        self.update_account_status()

        equity = self.assets['USDT']['wallet'] - self.assets['USDT']['debt']
        equity += (self.assets['BTC']['wallet'] - self.assets['BTC']['debt']) * mark_price

        target_btc = leverage * equity / mark_price

        order_size_btc = target_btc + self.assets['BTC']['debt'] - self.assets['BTC']['wallet']

        print(f"Order size BTC: leverage({leverage}) target_btc({target_btc}) order_size_btc({order_size_btc})")

        # Borrow
        borrow_extra = 1.01
        if leverage > 0:
            borrow_usdt = borrow_extra * order_size_btc * mark_price - self.assets['USDT']['wallet']
            if borrow_usdt > 0:
                print(f"Borrow USDT ({borrow_usdt})")
                transaction = self.client.create_margin_loan(asset='USDT', amount=str(borrow_usdt))
                print("Borrow USDT transaction", transaction)

        else:
            borrow_btc = borrow_extra * -order_size_btc - self.assets['BTC']['wallet']
            if borrow_btc > 0:
                print(f"Borrow BTC ({borrow_btc})")
                quit()
                transaction = self.client.create_margin_loan(asset='BTC', amount=str(borrow_btc))
                print("Borrow BTC result", transaction)

        # Trade BTC<->USDT
        if leverage > 0:
            order = self.client.create_margin_order(
                symbol='BTCUSDT',
                side=binance.enums.SIDE_BUY,
                type=binance.enums.ORDER_TYPE_MARKET,
                timeInForce=binance.enums.TIME_IN_FORCE_GTC,
                quantity=order_size_btc,
                newOrderRespType=binance.enums.ORDER_RESP_TYPE_RESULT)
            print(f"Order BTC ({order_size_btc})")
            print("Order BTC result", order)

        # Repay debt
        if leverage > 0 and self.assets['BTC']['debt'] > 0:
            if self.assets['BTC']['debt'] * mark_price > self.assets['USDT']['wallet']:
                print("Trying to repay BTC debt, not enough USDT funds!!!")
                return
            self.client.repay_margin_loan(asset='BTC', amount=str(self.assets['BTC']['debt']))

        elif leverage < 0 and self.assets['USDT']['debt'] > 0:
            if self.assets['USDT']['debt'] > self.assets['BTC']['wallet'] * mark_price:
                print("Trying to repay USDT debt, not enough BTC funds!!!")
                return
            self.client.repay_margin_loan(asset='USDT', amount=str(self.assets['USDT']['debt']))

        """
        target_btc_wallet = 0
        target_btc_debt = 0
        target_usdt_wallet = 0
        target_usdt_debt = 0

        if target_btc > 0:
            target_btc_wallet = target_btc
            if equity >= target_btc * mark_price:
                target_usdt_debt = 0
            else:
                target_usdt_debt = target_btc * mark_price - equity
        else:
            target_btc_debt = 0
            target_usdt_wallet = 0
        """

        return

