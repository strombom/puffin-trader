
chiffer = "ÖSNAGELRRATGSNIHAKUPELLINLKAIDTEIOFTFENNAMRTÄYHNKKGIIGTOUNOTRENNNCSPDKELEBSDVOAVMDFSMÄAGTASNTCEPÖDDORYEOGSHT"

n, m = 9, 12

for row in range(n):
    for col in range(m):
        print(chiffer[col + row * m] + ",", end='')
    print()

quit()





import numpy as np
import pyqtgraph.examples
from scipy.optimize import least_squares

pyqtgraph.examples.run()
quit()

lengths = np.array((5, 7, 9, 11, 13, 15, 19, 23, 27, 33, 39, 47, 57, 69, 83))
vols = np.array((0.001399389, 0.002204447, 0.002846914, 0.003408306, 0.003919525, 0.004379303, 0.005162534, 0.005829745, 0.006447528, 0.007344334, 0.008159027, 0.00904814, 0.010109587, 0.011172478, 0.012303446))

angs = np.array((0.005731540860194895, 0.005680738000689134, 0.006131365322804672, 0.00600933979017082, 0.005787305054981928, 0.004911227873430013, 0.0038157559315258327, 0.0034388036794658072, 0.0029515730520446315, 0.0029606684691220186, 0.002643517237611026, 0.0025361979809769286, 0.0022151843415383343, 0.0017697111153920364, 0.0016629634762057588))


def func(x):
    a = np.power(lengths * x[0], x[1])
    return a - 1 / angs


x0 = (1, 1)
res = least_squares(func, x0)
print(res.x)

a = 1 / np.power(lengths * res.x[0], res.x[1])
print(a)
quit()


def p(b, c, d, x, y):
    return ((b or c) and d) and ((x and y) or c)


tab = [(True, False, True, True, True),
       (False, False, True, True, True),

       (False, True, True, True, True),
       (False, False, True, True, True),

       (True, True, True, True, True),
       (True, True, False, True, True),

       (True, False, True, True, True),
       (True, False, True, False, True),

       (True, False, True, True, True),
       (True, False, True, True, False)
       ]

for t in tab:
    print(p(*t))
quit()







import json
from time import sleep

import binance.enums
import binance.exceptions
from binance.client import Client
from binance.websockets import BinanceSocketManager


with open('binance_account.json') as f:
    account_info = json.load(f)
    api_key = account_info['api_key']
    api_secret = account_info['api_secret']

client = Client(api_key, api_secret)


def process_depth_message(data):
    print("ask", data['asks'][0][0], "bid", data['bids'][0][0])


margin_socket_manager = BinanceSocketManager(client)
margin_socket_manager.start_depth_socket('BTCUSDT', process_depth_message, depth=BinanceSocketManager.WEBSOCKET_DEPTH_5)
margin_socket_manager.start()

while True:
    sleep(1)
