import asyncio
import json
#import ccxtpro
from decimal import Decimal

from cryptofeed import FeedHandler
from cryptofeed.callback import BookCallback, TradeCallback, CandleCallback, UserDataCallback, LastPriceCallback, BalancesCallback
from cryptofeed.defines import BID, ASK, BYBIT, L2_BOOK, TRADES, CANDLES, LAST_PRICE, BALANCES


class PhemexAccount:
    def __init__(self, api_key, api_secret, symbols):
        f = FeedHandler(config="cryptofeed_config.yaml")

        #f.add_feed(PHEMEX, channels=[L2_BOOK, TRADES, CANDLES],
        #           symbols=["ETH-USD-PERP", "BTC-USD-PERP"],
        #           callbacks={TRADES: TradeCallback(self.trade)})

        #f.add_feed(PHEMEX, channels=[LAST_PRICE], symbols=["ETH-USD-PERP", "BTC-USD-PERP"],
        #           callbacks={LAST_PRICE: LastPriceCallback(price)})

        #f.add_feed(PHEMEX, channels=[USER_DATA], symbols=["BTC-USD-PERP"],
        #           callbacks={USER_DATA: UserDataCallback(userdata)}, timeout=-1)

        #f.add_feed(PHEMEX, channels=[BALANCES], symbols=["BTC-USD-PERP"],
        #           callbacks={BALANCES: UserDataCallback(self._balance)}, timeout=-1)

        f.add_feed(BYBIT,
                   channels=[BALANCES],
                   symbols=["ETH-USDT-PERP", "DOGE-USDT-PERP"],
                   callbacks={BALANCES: BalancesCallback(self._balances)},
                   timeout=-1)

        f.run()

    async def _balances(self, feed, symbol, data, receipt_timestamp):
        print(f'{feed} Balance ({symbol}): {data}')

    async def _balance(self, feed, data: dict, receipt_timestamp):
        print(f'{feed} User data update: {data}')

    async def trade(self, feed, symbol, order_id, timestamp, side, amount, price, receipt_timestamp):
        assert isinstance(timestamp, float)
        assert isinstance(side, str)
        assert isinstance(amount, Decimal)
        assert isinstance(price, Decimal)
        print(f'{feed} Trades: {symbol}: Side: {side} Amount: {amount} Price: {price} Time: {timestamp}')

        """
        self.exchange = ccxtpro.phemex({
            'apiKey': api_key,
            'secret': api_secret
        })

        loop = asyncio.get_event_loop()
        coroutine = self.exchange.fetch_balance(params={"type": "swap", "code": "USD"})
        a = loop.run_until_complete(coroutine)

        print(a)

        #balance = self.exchange.fetch_balance(params={})

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        balance = loop.run_until_complete(self.exchange.fetch_balance(params={}))

        #balance = await self.exchange.fetch_balance(params={})
        print('balance', balance)
        print('watchBalance' in self.exchange.has)

        asyncio.run(self._watch_balance())
        """

    """
    async def _watch_balance(self):
        while True:
            try:
                balance = await self.exchange.watch_balance(params={})
                print(self.exchange.iso8601(self.exchange.milliseconds()), balance)
            except Exception as e:
                print("Phemex watch balance error", e)

    def get_total_equity_usdt(self):
        pass

    def get_balance(self, asset):
        pass

    def get_mark_price(self, symbol):
        pass

    def market_sell(self, symbol, volume):
        pass
    """


if __name__ == '__main__':
    with open('credentials.json') as f:
        credentials = json.load(f)

    all_symbols = ['XLMUSDT', 'VETUSDT', 'BCHUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
                   'TRXUSDT', 'DOGEUSDT', 'LINKUSDT', 'LTCUSDT', 'XRPUSDT', 'MATICUSDT',
                   'THETAUSDT', 'CHZUSDT', 'BTTUSDT', 'EOSUSDT', 'NEOUSDT', 'ETCUSDT']
    phemex_account = PhemexAccount(api_key=credentials['phemex']['api_key'], api_secret=credentials['phemex']['api_secret'], symbols=all_symbols)


"""
import sys
import math
import logging
import threading
from decimal import Decimal
import websocket
import _thread
import time

from cryptofeed import FeedHandler
from cryptofeed.defines import PHEMEX, USER_DATA, PERPETUAL, LAST_PRICE
from cryptofeed.callback import UserDataCallback, LastPriceCallback
from cryptofeed.symbols import Symbol


async def userdata(feed, data: dict, receipt_timestamp):
    print(f'{feed} User data update: {data}')


async def price(feed, symbol, last_price, receipt_timestamp):
    print(f'{feed} Price: {symbol}: {last_price}')


class PhemexAccount:
    def __init__(self, api_key, api_secret, symbols):
        _symbols = [Symbol('LTC', 'USD', type=PERPETUAL), Symbol('XTZ', 'USD', type=PERPETUAL)]
        f = FeedHandler(config="cryptofeed_config.yaml")
        #f.add_feed(PHEMEX, channels=[LAST_PRICE], symbols=["LTC-USD-PERP", "XTZ-USD-PERP"], callbacks={LAST_PRICE: LastPriceCallback(price)})
        f.add_feed(PHEMEX, channels=[USER_DATA], symbols=["LTC-USD-PERP", "XTZ-USD-PERP"], callbacks={USER_DATA: UserDataCallback(userdata)}, timeout=-1)
        f.run()


if __name__ == '__main__':

    def on_message(ws, message):
        print(message)


    def on_error(ws, error):
        print(error)


    def on_close(ws, close_status_code, close_msg):
        print("### closed ###")


    def on_open(ws):
        def run(*args):
            for i in range(3):
                time.sleep(1)
                ws.send("Hello %d" % i)
            time.sleep(1)
            ws.close()
            print("thread terminating...")

        _thread.start_new_thread(run, ())

    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://echo.websocket.org/",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    ws.run_forever()
"""
"""
print('a')
all_symbols = ['XLMUSDT', 'VETUSDT', 'BCHUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
               'TRXUSDT', 'DOGEUSDT', 'LINKUSDT', 'LTCUSDT', 'XRPUSDT', 'MATICUSDT',
               'THETAUSDT', 'CHZUSDT', 'BTTUSDT', 'EOSUSDT', 'NEOUSDT', 'ETCUSDT']
print('b')
phemex_account = PhemexAccount(api_key=1, api_secret=1, symbols=all_symbols)
print('c')
"""


"""
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
        self.update_balance()

        # Get all prices
        prices = self._client.get_all_tickers()
        for price in prices:
            if price['symbol'] in symbols:
                self._mark_price[price['symbol']] = float(price['price'])

        streams = [f"{symbol.lower()}@trade" for symbol in self._symbols]
        self.twm.start_multiplex_socket(callback=self._process_tick_message, streams=streams)
        self.twm.start_margin_socket(callback=self._process_margin_message)

    def _process_margin_message(self, data):
        # Todo: 2021-06-29 MATIC was showing a balance of 110 but was 30 on the exchange.
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

    def update_balance(self):
        with self._balance_lock:
            try:
                account_info = self._client.get_margin_account()
            except requests.exceptions.ConnectionError as e:
                logging.error(f"binance_account.update_balance error: {e}")
                return
            except:
                logging.error(f"binance_account.update_balance unexpected error: {sys.exc_info()[0]}")
                return
            for asset in account_info['userAssets']:
                if asset['asset'] + 'USDT' in self._symbols or asset['asset'] == 'USDT':
                    # print(f"Update balance {asset['free']} {asset['asset']}")
                    self._balance[asset['asset']] = float(asset['free'])
                    #self._debt[asset['asset']] = asset['borrowed']

    #def get_portfolio(self):
    #    portfolio = set()
    #    for trade_pair in self._balance:
    #        if self._balance[trade_pair] > 0:
    #            portfolio.add(trade_pair)
    #    return portfolio

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
            logging.info(f"Market buy {volume} {symbol} FAILED! Volume too low.")
            return {'quantity': 0, 'price': 0, 'error': 'low volume'}
        elif quantity < self._min_lot_size[symbol]:
            logging.info(f"Market buy {volume} {symbol} FAILED! Volume below min lot size: {self._min_lot_size[symbol]}.")
            return {'quantity': 0, 'price': 0, 'error': 'low volume'}
        elif quantity * self._mark_price[symbol] < self._min_notional[symbol]:
            logging.info(f"Market buy {volume} {symbol} FAILED! Price * quantity below min notional: {self._min_notional[symbol]}.")
            return {'quantity': 0, 'price': 0, 'error': 'low volume'}

        for retry in range(3):
            try:
                order = self._client.create_margin_order(
                    symbol=symbol,
                    side=binance.enums.SIDE_BUY,
                    type=binance.enums.ORDER_TYPE_MARKET,
                    quantity=quantity
                )

                if order['status'] != 'FILLED':
                    logging.info(f"Market buy  {quantity} {symbol} FAILED! {order}")
                    return {'quantity': 0, 'price': 0, 'error': 'not filled'}
                else:
                    fill_quantity = 0
                    fill_value = 0
                    for fill in order['fills']:
                        fill_quantity += float(fill['qty'])
                        fill_value += float(fill['qty']) * float(fill['price'])
                    price = fill_value / fill_quantity

                    logging.info(f"Market buy {quantity} {symbol} OK")
                    return {'quantity': float(order['executedQty']), 'price': price, 'error': 'ok'}

            except BinanceAPIException as e:
                logging.info(f"Market buy  {quantity} {symbol} error: {e}")

            except requests.exceptions.ConnectionError as e:
                logging.error(f"market_buy ConnectionError: {e}")

        logging.info(f"Market buy  {quantity} {symbol} FAILED!")
        return {'quantity': 0, 'price': 0, 'error': 'fail'}

    def market_sell(self, symbol, volume):
        balance = self._balance[symbol.replace('USDT', '')]
        volume = min(volume, balance)

        factor = 10 ** -self._tick_size[symbol]
        quantity = math.floor(volume * factor) / factor
        if quantity == 0:
            logging.info(f"Market sell {volume} {symbol} FAILED! Volume too low.")
            return {'quantity': 0, 'price': 0, 'error': 'low volume'}

        for retry in range(3):
            try:
                order = self._client.create_margin_order(
                    symbol=symbol,
                    side=binance.enums.SIDE_SELL,
                    type=binance.enums.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                if order['status'] != 'FILLED':
                    logging.info(f"Market sell {quantity} {symbol} FAILED! {order}")
                    return {'quantity': 0, 'price': 0, 'error': 'fail'}
                else:
                    fill_quantity = 0
                    fill_value = 0
                    for fill in order['fills']:
                        fill_quantity += float(fill['qty'])
                        fill_value += float(fill['qty']) * float(fill['price'])
                    price = fill_value / fill_quantity

                    logging.info(f"Market sell {quantity} @ {price} {symbol} OK, {order}")
                    return {'quantity': float(order['executedQty']), 'price': price, 'error': 'ok'}

            except BinanceAPIException as e:
                logging.info(f"Market sell  {quantity} {symbol} error: {e}")

            except requests.exceptions.ConnectionError as e:
                logging.error(f"market_buy ConnectionError: {e}")

        logging.info(f"Market sell  {quantity} {symbol} FAILED!")
        return {'quantity': 0, 'price': 0, 'error': 'fail'}

"""
