
import csv
import sys
import matplotlib.pyplot as plt
#from matplotlib.ticker import LinearLocator, MultipleLocator, FormatStrFormatter, AutoMinorLocator

sys.path.append("../Common")

from misc import string_to_datetime
from OrderBook import make_order_books



order_books = make_order_books(None, None)

filepath = 'alphaengine_trades.csv'
#¤filepath = 'trades.csv'

actions = {'dc_0': ([], []),
           'dc_1': ([], []),
           'dc_2': ([], []),
           'os_0': ([], []),
           'os_1': ([], []),
           'os_2': ([], []),
           'limit_order_buy': ([], []),
           'limit_order_sell': ([], []),
           'market_order_buy': ([], []),
           'market_order_sell': ([], []),
           'wallet': ([], []),
           'contracts': ([], []),
           'value': ([], []),
           'ask': ([], []),
           'bid': ([], []),
           'liquidity': ([], [])}

with open(filepath, 'r') as csv_file:
    prev_contracts = 0
    prev_wallet = 1
    prev_value = 0

    for x, row in enumerate(csv.reader(csv_file)):
        timestamp = string_to_datetime(row[0], fmt='%Y-%m-%d %H:%M:%S')
        action_name = row[1]
        if 'RunnerEvent' in action_name:
            #timestamp =

            if row[6] == 'False':
                continue

            runner_idx = int(row[2])
            ask, bid = float(row[3]), float(row[4])
            if 'up' in action_name:
                price = ask
            else:
                price = bid
            if 'direction_change' in action_name:
                key = 'dc_'
            else:
                key = 'os_'
            key += str(runner_idx)

            actions[key][0].append(timestamp)
            actions[key][1].append(price)

            actions['liquidity'][0].append(timestamp)
            actions['liquidity'][1].append(float(row[5]))

        elif 'limit_' in action_name or 'market_' in action_name:
            price = float(row[3])

            actions['contracts'][0].append(timestamp)
            actions['contracts'][1].append(prev_contracts)
            actions['contracts'][0].append(timestamp)
            actions['contracts'][1].append(float(row[4]))
            prev_contracts = float(row[4])

            actions['wallet'][0].append(timestamp)
            actions['wallet'][1].append(prev_wallet)
            actions['wallet'][0].append(timestamp)
            actions['wallet'][1].append(float(row[5]))
            prev_wallet = float(row[5])

            if prev_value == 0:
                prev_value = price * prev_wallet
            actions['value'][0].append(timestamp)
            actions['value'][1].append(prev_value)
            actions['value'][0].append(timestamp)
            actions['value'][1].append(float(row[6]))
            prev_value = float(row[6])

            if 'limit_' in action_name:
                key = 'limit_order'
            else:
                key = 'market_order'
            if 'buy' in action_name:
                key += '_buy'
            else:
                key += '_sell'

            actions[key][0].append(timestamp)
            actions[key][1].append(price)

for idx, order_book in enumerate(order_books):
    actions['ask'][0].append(order_book.timestamp)
    actions['ask'][1].append(order_book.ask)
    actions['bid'][0].append(order_book.timestamp)
    actions['bid'][1].append(order_book.bid)

#for key in actions:
#    print(key, actions[key])
#    #print(actions[key][1])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#ax1 = plt.subplot(211)
#ax2 = plt.subplot(212, sharex=ax1)

ax1.plot(actions['ask'][0], actions['ask'][1])
ax1.plot(actions['bid'][0], actions['bid'][1])

ax1.scatter(actions['market_order_buy'][0], actions['market_order_buy'][1], marker='P')
ax1.scatter(actions['market_order_sell'][0], actions['market_order_sell'][1], marker='X')
ax1.scatter(actions['limit_order_buy'][0], actions['limit_order_buy'][1], marker='+')
ax1.scatter(actions['limit_order_sell'][0], actions['limit_order_sell'][1], marker='x')

for key in ['dc_0', 'dc_1', 'dc_2']:
    action = actions[key]
    #ax1.plot(action[0], action[1])

for key in ['os_0', 'os_1', 'os_2']:
    action = actions[key]
    ax1.scatter(action[0], action[1])


ax2.set_ylabel('wallet', color='green')
ax2.plot(actions['wallet'][0], actions['wallet'][1], color='green')

ax2b = ax2.twinx()
ax2b.set_ylabel('contracts', color='blue')
ax2b.plot(actions['contracts'][0], actions['contracts'][1], color='blue')

ax2c = ax2.twinx()
ax2c.set_ylabel('value', color='red')
ax2c.plot(actions['value'][0], actions['value'][1], color='red')

ax2d = ax2.twinx()
ax2d.set_ylabel('value', color='orange')
ax2d.plot(actions['liquidity'][0], actions['liquidity'][1], color='orange')

#fig, ax=plt.subplots(num=10, clear=True)
plt.show()

quit()

