
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
    def __init__(self, pos_price):
        self.wallet_btc = 1.0
        self.pos_price = pos_price
        self.pos_contracts = 0.0

    def get_position_direction(self):
        if self.pos_contracts >= 0:
            return 'long'
        else:
            return 'short'

def execute_order(account, price, side):



    """
    // Fee
    wallet -= fee * abs(order_contracts / price);
        
    // Realised profit and loss
    // Wallet only changes when abs(contracts) decrease
    if (pos_contracts > 0 && order_contracts < 0) {
        wallet += (1 / pos_price - 1 / price) * std::min(-order_contracts, pos_contracts);
    }
    else if (pos_contracts < 0 && order_contracts > 0) {
        wallet += (1 / pos_price - 1 / price) * std::max(-order_contracts, pos_contracts);
    }

    // Calculate average entry price
    if ((pos_contracts >= 0 && order_contracts > 0) || 
        (pos_contracts <= 0 && order_contracts < 0)) {
        pos_price = (pos_contracts * pos_price + order_contracts * price) / (pos_contracts + order_contracts);
    }
    else if ((pos_contracts >= 0 && (pos_contracts + order_contracts) < 0) || 
             (pos_contracts <= 0 && (pos_contracts + order_contracts) > 0)) {
        pos_price = price;
    }

    // Calculate position contracts
    pos_contracts += order_contracts;
    """


fee = 0.00075
stop_loss = 0.005
min_profit = 0.0025

account = Account(events_price[0])

event_type = 'top'
for idx in range(len(events_x)):

    execute = False
    execution_price = 0.0
    price = events_price[idx]
    pos_direction = account.get_position_direction()

    if event_type == 'top':
        if pos_direction == 'long':
            min_profit_price = account.pos_price * (1 + min_profit)
            if price > min_profit_price:
                execute_order(account, events_offset[idx], 'short')
            print("min_profit_price", min_profit_price)
            pass

        elif pos_direction == 'short':
            pass

    elif event_type == 'bot':
        if pos_direction == 'long':
            pass

        elif pos_direction == 'short':
            pass

    print("event_type", event_type)
    print("pos_direction", pos_direction)
    print("price", price)
    quit()


    print(idx, events_price[idx], events_trig_price[idx], event_type)
    quit()


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
