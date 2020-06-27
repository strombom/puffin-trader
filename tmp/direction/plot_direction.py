
import numpy as np
import csv

xs = []
prices = []

with open('C:\\development\\github\\puffin-trader\\tmp\\direction\\data' + '.csv') as csvfile:
    reader = csv.reader(csvfile)
    x = 0
    for row in reader:
        price = float(row[0])
        prices.append(price)
        xs.append(x)
        x = x + 1

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
