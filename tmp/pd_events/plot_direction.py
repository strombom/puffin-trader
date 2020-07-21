
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import numpy as np
import csv


intervals = []
with open('intervals.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        intervals.append(float(row[0]))
intervals = np.array(intervals)

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

ticks_x = []
ticks_price = []
with open('ticks.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        ticks_x.append(float(row[0]))
        ticks_price.append(float(row[1]))
ticks_x = np.array(ticks_x) - 1
ticks_price = np.array(ticks_price)

obt_x = []
obt_price = []
with open('orderbook_top.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        obt_x.append(float(row[0]))
        obt_price.append(float(row[1]))
obt_x = np.array(obt_x) - 1
obt_price = np.array(obt_price)

obb_x = []
obb_price = []
with open('orderbook_bot.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        obb_x.append(float(row[0]))
        obb_price.append(float(row[1]))
obb_x = np.array(obb_x) - 1
obb_price = np.array(obb_price)



fig, axs = plt.subplots(nrows=1)
ax = axs
ax.plot(intervals, color='green', linewidth=0.6)
#ax.plot(xs, orderbook, '-', linewidth=0.6, markersize=2, color='green')
#ax.plot(xs, orderbook - 0.5, '-', linewidth=0.6, markersize=2, color='red')
ax.scatter(events_x, events_price, color='blue', s=16.7)
ax.scatter(events_trig_x, events_trig_price, color='red', s=10.7)
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
