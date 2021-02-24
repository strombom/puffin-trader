import json
from binance.client import Client
from binance.websockets import BinanceSocketManager


class BinanceAccount:
    assets = {}

    def __init__(self):
        with open('binance_account.json') as f:
            account_info = json.load(f)
            api_key = account_info['api_key']
            api_secret = account_info['api_secret']

        self.client = Client(api_key, api_secret)
        self.socket_manager = BinanceSocketManager(self.client)

        def process_margin_message(data):
            print("mm", data)

        self.socket_manager.start_margin_socket(process_margin_message)
        self.socket_manager.start()

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
            self.assets[name] = {'balance': float(asset['free']),
                                 'borrowed': float(asset['borrowed'])}

