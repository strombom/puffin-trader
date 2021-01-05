import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from matplotlib import markers
from multiprocessing.connection import Client

from BitmexSim.bitmex_simulator import BitmexSimulator
from Common.Misc import timestamp_to_string
from Indicators.supersmoother import SuperSmoother
from ie_scatter import make_ie_scatters

with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
    delta, order_books, runner = pickle.load(f)

p = np.array(runner.ie_prices)
runner.ie_prices = p + 0


ie_scatters, ie_overshoots = make_ie_scatters(runner)

pos_x = 20
start_x = 0

c_x = np.arange(0, pos_x - start_x)
prices = runner.ie_prices[start_x:pos_x]
r = stats.linregress(c_x, prices)
c_y = c_x * r.slope + r.intercept

address = ('localhost', 27567)
conn = Client(address, authkey=b'secret password')
conn.send((ie_scatters, ie_overshoots, None))
conn.close()
quit()

print(c_x)
print(prices)
print(r)
print(c_y)
#quit()

ax1 = plt.subplot(1, 1, 1)
for scatter_idx in range(len(ie_scatters)):
    ie_scatter = ie_scatters[scatter_idx]
    ax1.scatter(ie_scatter['x'], ie_scatter['y'], marker=ie_scatter['marker'], color=ie_scatter['color'], s=ie_scatter['size'])

ax1.scatter(ie_overshoots['x'], ie_overshoots['y'], marker='_', color='xkcd:grey', s=40)

ax1.plot(c_x, c_y, label='C')

#for smooth in smooths:
#    ax1.plot(smooths[smooth])

#ax2 = plt.subplot(3, 1, 2, sharex=ax1)
#ax2.plot(values)

#ax3 = plt.subplot(3, 1, 3, sharex=ax1)
#ax3.plot(leverages)

#plt.plot(runner.os_times, runner.os_prices, label=f'OS')
plt.show()






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


quit()

ax1 = plt.subplot(1, 1, 1)
ax1.grid(True)
# plt.plot(times, prices, label=f'price')
# plt.plot(times, asks, label=f'ask')
# plt.plot(times, bids, label=f'bid')
# plt.plot(runner.os_times, os_prices, label=f'OS')
# plt.scatter(runner.os_times, os_prices, label=f'OS', s=5 ** 2)
# plt.scatter(runner.dc_times, dc_prices, label=f'DC', s=7 ** 2)
# plt.scatter(runner.ie_times, ie_prices, label=f'IE', s=5 ** 2)

x = np.arange(0, len(ie_prices))

plt.plot(x, ie_prices, label=f'IE')
plt.scatter(x, ie_prices, label=f'IE')

plt.legend()

plt.show()
