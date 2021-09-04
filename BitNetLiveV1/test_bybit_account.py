import json
import pytest

from bybit_account import BybitAccount


all_symbols = ['ADAUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'BTTUSDT', 'CHZUSDT', 'DOGEUSDT', 'EOSUSDT', 'ETCUSDT',
               'ETHUSDT', 'LINKUSDT', 'LTCUSDT', 'MATICUSDT', 'NEOUSDT', 'THETAUSDT', 'TRXUSDT', 'VETUSDT',
               'XLMUSDT', 'XRPUSDT']
with open('credentials.json') as f:
    credentials = json.load(f)
bybit_account = BybitAccount(
    api_key=credentials['bybit']['api_key'],
    api_secret=credentials['bybit']['api_secret'],
    symbols=all_symbols
)


def test_get_mark_price_usdt():
    assert bybit_account.get_mark_price('BTCUSDT') != 0


def test_get_balance_usdt():
    assert bybit_account.get_balance('USDT') != 0


def test_get_balance_btcusdt():
    assert bybit_account.get_balance('THETAUSDT') != 0




