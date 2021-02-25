import json
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

        print(f"ask ({self.mark_price_ask}), bid ({self.mark_price_bid})")
        print("ok")
        quit()
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

        self.order(0.5)

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
        print(f"Order leverage({leverage})")
        order_size_btc = self.calculate_order_size_btc(leverage=leverage)
        print(f"Order size BTC {order_size_btc}")

        # Repay debt
        if order_size_btc > 0:
            if self.assets['BTC']['debt'] > 0:
                if self.assets['BTC']['wallet'] > self.assets['BTC']['debt']:
                    print("Repay debt BTC ", self.assets['BTC']['debt'])
                    transaction = self.client.repay_margin_loan(asset='BTC', amount=str(self.assets['BTC']['debt']))
                    print("Repay debt transaction ", transaction)

                    self.wallet_btc -= self.debt_btc
                    self.debt_btc = 0
                else:
                    self.debt_btc -= self.wallet_btc
                    self.wallet_btc = 0

        elif order_size_btc < 0:
            if self.debt_usdt > 0:
                if self.wallet_usdt > self.debt_usdt:
                    self.wallet_usdt -= self.debt_usdt
                    self.debt_usdt = 0
                else:
                    self.debt_usdt -= self.wallet_usdt
                    self.wallet_usdt = 0

        self.update_account_status()

    def calculate_order_size_btc(self, leverage):
        order_size_btc = 0

        if leverage > 0:
            if self.account['BTC']['debt'] > 0:
                order_size_btc += self.account['BTC']['debt']

        elif leverage < 0:
            if self.account['BTC']['wallet'] > 0:
                order_size_btc -= self.account['BTC']['wallet']

        equity = self.account['USDT']['wallet'] - self.account['USDT']['debt']
        equity += (self.account['BTC']['wallet'] - self.account['BTC']['debt'])

        print(f"calculate order size btc leverage({leverage})")
        quit()

        return order_size_btc