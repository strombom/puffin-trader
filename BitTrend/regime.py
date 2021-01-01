import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers

from BitmexSim.bitmex_simulator import BitmexSimulator
from Common.Misc import timestamp_to_string
from Indicators.supersmoother import SuperSmoother
from IntrinsicTime.runner import Direction


with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
    delta, order_books, runner = pickle.load(f)

#ie_prices = np.array(runner.ie_prices)
#ie_prices = np.log(ie_prices)

x = np.arange(0, len(runner.ie_times))
ie_directions = np.empty(len(runner.ie_times))

scatters = []
colors = ['xkcd:green', 'xkcd:light red', 'xkcd:lime', 'xkcd:baby pink']
for i in range(4):
    marker = '+'  # '^' if i % 2 == 0 else 'v'
    size = 40
    color = colors[i]
    scatters.append({'marker': marker, 'size': size, 'color': color, 'x': [], 'y': []})

overshoots = {'x': [], 'y': []}
idx_dc, idx_os = 0, 0
direction = Direction.down
for idx_ie, timestamp in enumerate(runner.ie_times):
    turn = False
    while idx_os + 1 < len(runner.os_times) and timestamp > runner.os_times[idx_os + 1]:
        idx_os += 1
        overshoots['x'].append(idx_ie - 1)
        overshoots['y'].append(runner.os_prices[idx_os])
        turn = True
        if runner.ie_prices[idx_ie] > runner.os_prices[idx_os]:
            direction = Direction.up
        else:
            direction = Direction.down

    while idx_dc + 1 < len(runner.dc_times) and timestamp >= runner.dc_times[idx_dc + 1]:
        idx_dc += 1
    if idx_dc >= len(runner.dc_times):
        break

    if direction == Direction.up:
        scatter_idx = 0
    else:
        scatter_idx = 1

    if idx_ie + 1 < len(runner.ie_times):
        if runner.ie_times[idx_ie + 1] == runner.ie_times[idx_ie]:
            scatter_idx += 2  # Free fall

    scatters[scatter_idx]['x'].append(idx_ie)
    scatters[scatter_idx]['y'].append(runner.ie_prices[idx_ie])

smooth_periods = [100, 500]
smooths = {}
for smooth_period in smooth_periods:
    smooth = []
    smoother = SuperSmoother(period=smooth_period, initial_value=runner.ie_prices[0])
    for price in runner.ie_prices:
        smooth.append(smoother.append(price))
    smooths[smooth_period] = smooth

values = []
leverages = []
sim = BitmexSimulator(max_leverage=2.0, mark_price=runner.ie_prices[0])
direction = Direction.up
for idx in range(len(runner.ie_times) - 1):
    mark_price = runner.ie_prices[idx]

    order_size = 0.0

    if direction == Direction.down and smooths[100][idx] > smooths[500][idx]:
        direction = Direction.up
        order_size = sim.calculate_order_size(leverage=4.0, mark_price=mark_price)

    elif direction == Direction.up and smooths[100][idx] < smooths[500][idx]:
        direction = Direction.down
        order_size = sim.calculate_order_size(leverage=-5.0, mark_price=mark_price)

    if order_size != 0:
        sim.market_order(order_contracts=order_size, mark_price=mark_price)

    values.append(sim.get_value(mark_price=mark_price))
    leverages.append(sim.get_leverage(mark_price=mark_price))


ax1 = plt.subplot(3, 1, 1)
for scatter_idx in range(len(scatters)):
    scatter = scatters[scatter_idx]
    ax1.scatter(scatter['x'], scatter['y'], marker=scatter['marker'], color=scatter['color'], s=scatter['size'])

ax1.scatter(overshoots['x'], overshoots['y'], marker='_', color='xkcd:grey', s=40)
for smooth in smooths:
    ax1.plot(smooths[smooth])

ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.plot(values)

ax3 = plt.subplot(3, 1, 3, sharex=ax1)
ax3.plot(leverages)

#plt.plot(runner.os_times, runner.os_prices, label=f'OS')
plt.show()







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
