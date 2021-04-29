import json
import time
import logging
from collections import deque

import zmq
from binance.client import Client


def get_historic_data(config: dict, runners: dict, ie_prices: dict, last_idx: int):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(config['binance_kline_server_address'])
    socket.send_pyobj(("get_since", -30))
    message = socket.recv_pyobj()
    print(message['last_idx'], len(message['mark_prices']))


def trader():
    with open('binance_account.json') as f:
        account_info = json.load(f)
        api_key = account_info['api_key']
        api_secret = account_info['api_secret']

    with open('config.json') as f:
        config = json.load(f)

    runners, ie_prices = {}, {}
    last_idx = 0

    get_historic_data(config=config, runners=runners, ie_prices=ie_prices, last_idx=last_idx)
    lengths = config['lengths']
    print(lengths)
    quit()

    logging.info("Start Binance client")
    binance_client = Client(api_key, api_secret)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.Formatter.converter = time.gmtime
    trader()
