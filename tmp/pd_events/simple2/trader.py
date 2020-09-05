
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import numpy as np
import csv


events_x = []
events_price = []
with open('events.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        events_x.append(float(row[0]))
        events_price.append(float(row[1]))
events_x = np.array(events_x) - 1
events_price = np.array(events_price)

events_trig_x = []
events_trig_price = []
with open('events_offset.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        events_trig_x.append(float(row[0]))
        events_trig_price.append(float(row[1]))
events_trig_x = np.array(events_trig_x) - 1
events_trig_price = np.array(events_trig_price)

print(events_x)
print(events_price)


class Account:
    def __init__(self):
        self.btc = 1.0
        self.usd = 0.0

class Position:
    def __init__(self, price, direction):
        self.price = price
        self.direction = direction


account = Account()
position = Position(events_price[0], "long")

fee = 0.00075
stop_loss = 0.005
min_profit = 0.0025

direction = "long"
for idx in range(len(events_x)):

    execute = False
    execution_price = 0.0
    if direction == "long" and position.direction = "short":
        pass

    elif direction == "short" and position.direction = "long":
        pass


    print(idx, events_price[idx], events_trig_price[idx], direction)
    quit()

    if direction == "long":
        direction = "short"
    else:
        direction = "long"


quit()



fig, axs = plt.subplots(nrows=1)
ax = axs
ax.scatter(events_x, events_price, color='blue', s=16.7)
ax.scatter(events_trig_x, events_trig_price, color='red', s=10.7)
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
