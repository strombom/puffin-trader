
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import numpy as np
import csv


episode = 1

filename = "rl\\episode_log_" + str(episode) + ".csv"

print(filename)

mark_prices = []
timestamps = []
account_values = []
pos_prices = []
pos_directions = []
pos_stop_loss = []
pos_leverages = []

first_row = True
with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if first_row:
            first_row = False
        else:
            mark_prices.append(float(row[0]))
            timestamps.append(float(row[1]))
            account_values.append(float(row[2]))
            pos_prices.append(float(row[3]))
            pos_directions.append(float(row[4]))
            pos_stop_loss.append(float(row[5]))
            pos_leverages.append(float(row[6]))

mark_prices = np.array(mark_prices)
timestamps = np.array(timestamps)
account_values = np.array(account_values)
pos_prices = np.array(pos_prices)
pos_directions = np.array(pos_directions)
pos_stop_loss = np.array(pos_stop_loss)
pos_leverages = np.array(pos_leverages)

fig, axs = plt.subplots(nrows=3)
ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]
ax1.plot(mark_prices, color='green', linewidth=0.6)
ax2.plot(account_values, color='green', linewidth=0.6)
ax3.plot(pos_leverages, color='green', linewidth=0.6)

plt.show()
quit()




fig, axs = plt.subplots(nrows=1)
ax = axs
ax.plot(intervals, color='green', linewidth=0.6)
#ax.plot(xs, orderbook, '-', linewidth=0.6, markersize=2, color='green')
#ax.plot(xs, orderbook - 0.5, '-', linewidth=0.6, markersize=2, color='red')
ax.scatter(events_x, events_price, color='blue', s=18.7)
ax.scatter(events_trig_x, events_trig_price, color='red', s=8.7)
ax.plot(ticks_x, ticks_price, color='blue', linewidth=0.4)
ax.plot(obt_x, obt_price, color='red', linewidth=0.4)
ax.plot(obb_x, obb_price, color='green', linewidth=0.4)
plt.show()




quit()

xs = np.array(xs)
prices = np.array(prices)
dirs = np.zeros(xs.size, dtype=np.int)
orderbook = np.empty(xs.size)

dirs[xs.size - 1] = 2

# Direction
#  0 - No direction
#  1 - Buy
#  2 - Sell

# Make orderbook
orderbook[0] = prices[0]
for i in range(1, xs.size):
    if prices[i] > orderbook[i - 1]:
        orderbook[i] = prices[i]
    elif prices[i] < orderbook[i - 1] - 0.5:
        orderbook[i] = prices[i] + 0.5
    else:
        orderbook[i] = orderbook[i - 1]

for i in range(1, xs.size):
    if prices[i] > orderbook[i - 1]:
        dirs[i - 1] = 2
    elif prices[i] < orderbook[i - 1] - 0.5:
        dirs[i - 1] = 1

last_direction = 1
for i in range(xs.size - 1, -1, -1):
    if dirs[i] == 0:
        dirs[i] = last_direction
    else:
        last_direction = dirs[i]
print(dirs)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

cmap = matplotlib.colors.ListedColormap(['green','red']) #'gray',

fig, axs = plt.subplots(nrows=1)
ax = axs
ax.plot(xs, orderbook, '-', linewidth=0.6, markersize=2, color='green')
ax.plot(xs, orderbook - 0.5, '-', linewidth=0.6, markersize=2, color='red')
ax.scatter(xs, prices, c=dirs, s=22.5, cmap=cmap)
plt.show()
