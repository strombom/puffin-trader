
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers

from Common.Misc import timestamp_to_string
from IntrinsicTime.runner import Direction


with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
    delta, order_books, runner = pickle.load(f)

os_prices = np.array(runner.os_prices)
os_prices = np.log(os_prices)

dc_prices = np.array(runner.dc_prices)
dc_prices = np.log(dc_prices)

ie_prices = np.array(runner.ie_prices)
ie_prices = np.log(ie_prices)

"""
x = 0
for ie_idx in range(len(ie_prices)):
    print(ie_idx)
    quit()
"""

x = np.arange(0, len(runner.ie_times))
#print(x)
#quit()
ie_directions = np.empty(len(runner.ie_times))


scatters = []
colors = ['xkcd:green', 'xkcd:light red', 'xkcd:lime', 'xkcd:baby pink']
for i in range(4):
    marker = '+'  # '^' if i % 2 == 0 else 'v'
    size = 40
    color = colors[i]
    #size = 35 if i // 2 % 2 == 0 else 75
    #color = 'xkcd:blue' if i // 4 % 2 == 0 else 'xkcd:light blue'
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

    scatter_idx = 0

    # Direction
    if direction == Direction.down:
        scatter_idx += 1

    # Turn
    #if turn:
    #    scatter_idx += 2

    # Free fall
    if idx_ie + 1 < len(runner.ie_times):
        if runner.ie_times[idx_ie + 1] == runner.ie_times[idx_ie]:
            scatter_idx += 2

    scatters[scatter_idx]['x'].append(idx_ie)
    #scatters[scatter_idx]['x'].append(runner.ie_times[idx_ie])
    scatters[scatter_idx]['y'].append(runner.ie_prices[idx_ie])

    #ie_directions[ie_idx]
    #print(f'Time({timestamp_to_string(timestamp)}) {direction}')

for scatter_idx in range(len(scatters)):
    scatter = scatters[scatter_idx]
    print(scatter['marker'], scatter['size'], scatter['color'])

    plt.scatter(scatter['x'], scatter['y'], marker=scatter['marker'], color=scatter['color'], s=scatter['size'])

plt.scatter(overshoots['x'], overshoots['y'], marker='_', color='xkcd:grey', s=40)

#plt.plot(runner.os_times, runner.os_prices, label=f'OS')
plt.show()

quit()


# # Measured direction
# for idx_runner, runner in enumerate(runners):
#     direction = Direction.up
#     idx_dc = 0
#     for idx_clock, timestamp in enumerate(runner_clock.ie_times):
#         while idx_dc < len(runner.dc_times) and runner.dc_times[idx_dc] < timestamp:
#             idx_dc += 1
#             if direction == Direction.up:
#                 direction = Direction.down
#             else:
#                 direction = Direction.up
#         if idx_dc >= len(runner.dc_times):
#             break
#         if direction == Direction.up:
#             measured_direction[idx_runner, idx_clock] = 1
#         else:
#             measured_direction[idx_runner, idx_clock] = 0



print(f'delta: {delta}')

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
