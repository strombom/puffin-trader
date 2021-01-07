
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from multiprocessing.connection import Client

from ie_scatter import make_ie_scatters

with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
    delta, order_books, runner = pickle.load(f)

p = np.array(runner.ie_prices)
runner.ie_prices = p + 0

ie_scatters, ie_overshoots = make_ie_scatters(runner)

data_buffer = []


def send_data():
    address = ('localhost', 27567)
    conn = Client(address, authkey=b'secret password')
    conn.send(data_buffer)
    conn.close()


def append_data(name, data):
    data_buffer.append({'name': name, 'data': data})


def live_plot(name, data):
    append_data(name, data)


def clear_lines():
    append_data('clear', None)


def annotate(x, y, text):
    append_data(f'annotate_{x}', {'x': x, 'y': y, 'text': text})


clear_lines()
# live_plot('ie_overshoots', ie_overshoots)
live_plot('ie_scatters', ie_scatters)


def channel(start, stop, width, name):
    start_x = start
    pos_x = stop

    c_x = np.arange(start_x, pos_x)
    prices = runner.ie_prices[start_x:pos_x]
    r = stats.linregress(c_x, prices)

    c_x = np.array([start_x, pos_x - 1])
    c_y = c_x * r.slope + r.intercept

    live_plot(f'channel_{name}_top', {'x': c_x, 'y': c_y, 'color': 'xkcd:blue'})
    live_plot(f'channel_{name}_mid', {'x': c_x, 'y': c_y + width, 'color': 'xkcd:green'})
    live_plot(f'channel_{name}_bot', {'x': c_x, 'y': c_y - width, 'color': 'xkcd:red'})


channel(0, 25, 22, 'a')
channel(20, 52, 18, 'b')
annotate(55, 9154, 'breakout')

send_data()

"""
smooth_periods = [600, 900]
smooths = {}
for smooth_period in smooth_periods:
    smooth = []
    smoother = SuperSmoother(period=smooth_period, initial_value=runner.ie_prices[0])
    for price in runner.ie_prices:
        smooth.append(smoother.append(price))
    smooths[smooth_period] = smooth

values = []
leverages = []
sim = BitmexSimulator(max_leverage=10.0, mark_price=runner.ie_prices[0])
direction = Direction.up
for idx in range(len(runner.ie_times) - 1):
    mark_price = runner.ie_prices[idx]

    order_size = 0.0

    if direction == Direction.down and smooths[smooth_periods[0]][idx] > smooths[smooth_periods[1]][idx]:
        direction = Direction.up
        order_size = sim.calculate_order_size(leverage=4.0, mark_price=mark_price)

    elif direction == Direction.up and smooths[smooth_periods[0]][idx] < smooths[smooth_periods[1]][idx]:
        direction = Direction.down
        order_size = sim.calculate_order_size(leverage=-5.0, mark_price=mark_price)

    if order_size != 0:
        sim.market_order(order_contracts=order_size, mark_price=mark_price)

    values.append(sim.get_value(mark_price=mark_price))
    leverages.append(sim.get_leverage(mark_price=mark_price))
"""

# ax1 = plt.subplot(1, 1, 1)
# ax1.grid(True)
# plt.plot(times, prices, label=f'price')
# plt.plot(times, asks, label=f'ask')
# plt.plot(times, bids, label=f'bid')
# plt.plot(runner.os_times, os_prices, label=f'OS')
# plt.scatter(runner.os_times, os_prices, label=f'OS', s=5 ** 2)
# plt.scatter(runner.dc_times, dc_prices, label=f'DC', s=7 ** 2)
# plt.scatter(runner.ie_times, ie_prices, label=f'IE', s=5 ** 2)

# x = np.arange(0, len(ie_prices))
#
# plt.plot(x, ie_prices, label=f'IE')
# plt.scatter(x, ie_prices, label=f'IE')
#
# plt.legend()
#
# plt.show()
