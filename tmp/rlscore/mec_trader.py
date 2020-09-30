
import sys
import csv
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


filename = "../PD_Events/events.csv"


maker_fee = -0.00025
taker_fee = 0.00075
max_leverage = 2.0


class Event:
    def __init__(self, timestamp, price):
        self.timestamp = (1596240000000 + int(timestamp)) / 1000
        self.price = price

    def __repr__(self):
        return f"Event({self.timestamp}, {self.price})"


events = []
with open(filename) as csv_file:
    for row in csv.reader(csv_file):
        ts = int(row[0])
        #if 1584000000000 < 1577836800000 + ts < 1584080000000:
        #    events.append(Event(ts, float(row[1])))

        #if 1577836800000 + ts < 1584006800000 or 1577836800000 + ts > 1584081300000:
        #    events.append(Event(ts, float(row[1])))

        #if 1596240000000 + ts > 1596376000000:
        events.append(Event(ts, float(row[1])))

        #if len(events) == 1000:
        #    break

#events = events[-3000:]

timestamp_start = datetime.fromtimestamp(events[0].timestamp)
timestamp_end = datetime.fromtimestamp(events[-1].timestamp)
print(timestamp_start)
print(timestamp_end)


class Position:
    def __init__(self, mark_price, initial_leverage, take_profit, min_profit):
        self.wallet = 1.0
        self.price = mark_price
        self.contracts = initial_leverage * self.wallet * self.price
        self.take_profit = take_profit
        self.min_profit = min_profit
        self._update()

    def _update(self):
        self.direction = self.contracts / abs(self.contracts)
        self.take_profit_price = self.price * (1 + self.direction * take_profit)
        self.stop_loss_price = self.price * (1 - self.direction * stop_loss)

        if self.wallet < 0.2:
            self.wallet = 0
            self.contracts = 0
            return

    def get_value(self, mark_price):
        if self.wallet == 0:
            return 0
        sign = self.contracts / abs(self.contracts)
        entry_value = abs(self.contracts / self.price)
        mark_value = abs(self.contracts / mark_price)
        upnl = sign * (mark_value - entry_value)
        return (self.wallet + upnl) * mark_price

    def market_order(self, order_leverage, mark_price):

        # print('market_order', wallet, pos_price, pos_contracts, order_contracts, mark_price)
        if self.wallet == 0:
            return

        #sign = self.contracts / abs(self.contracts)
        #entry_value = abs(self.contracts / self.price)
        #mark_value = abs(self.contracts / mark_price)
        #upnl = sign * (mark_value - entry_value)

        upnl = (1 / self.price - 1 / mark_price) * self.contracts

        max_contracts = max_leverage * (self.wallet + upnl) * mark_price
        margin = self.wallet * min(max(order_leverage, -max_leverage), max_leverage)
        order_contracts = min(max(margin * mark_price, -max_contracts), max_contracts) - self.contracts

        # Fee
        self.wallet -= taker_fee * abs(order_contracts / mark_price)

        # Realised profit and loss
        # Wallet only changes when abs(contracts) decrease
        if (self.contracts > 0) and (order_contracts < 0):
            self.wallet += (1 / self.price - 1 / mark_price) * min(-order_contracts, self.contracts)
        elif (self.contracts < 0) and (order_contracts > 0):
            self.wallet += (1 / self.price - 1 / mark_price) * max(-order_contracts, self.contracts)

        # Calculate average entry price
        if (self.contracts >= 0 and order_contracts > 0) or (self.contracts <= 0 and order_contracts < 0):
            self.price = (self.contracts * self.price + order_contracts * mark_price) / (self.contracts + order_contracts)
        elif (self.contracts >= 0 and self.contracts + order_contracts < 0) or (self.contracts <= 0 and self.contracts + order_contracts > 0):
            self.price = mark_price

        # Calculate position contracts
        self.contracts += order_contracts
        self._update()


def simulate(start_idx, end_idx, stop_loss, take_profit):
    volatility = 0.01
    volatilities = []
    vola_prices = []
    values = []
    drawdown = 0.0
    peak_price = events[0].price
    position = Position(events[0].price, max_leverage, stop_loss, take_profit)
    for idx in range(start_idx, end_idx):
        leverage = volatility * 60
        event = events[idx]
        if position.direction > 0 and event.price < position.stop_loss_price:
            position.market_order(-leverage, position.stop_loss_price)
        elif position.direction < 0 and event.price > position.stop_loss_price:
            position.market_order(leverage, position.stop_loss_price)

        if position.direction > 0 and event.price > position.take_profit_price:
            position.market_order(-leverage, event.price)
        elif position.direction < 0 and event.price < position.take_profit_price:
            position.market_order(leverage, event.price)

        peak_price = max(peak_price, event.price)
        if event.price < peak_price:
            drawdown = max(drawdown, (peak_price - event.price) / peak_price)
        values.append(position.wallet * event.price) #  position.get_value(event.price))

        vola_prices.append(event.price)
        if len(vola_prices) > 250:
            vola_prices = vola_prices[1:]
        volatility = max(vola_prices) / min(vola_prices) - 1
        volatilities.append(volatility)

        # print(f"go long, value({position.wallet * event.price} price({event.price})")
    return position, values, drawdown, volatilities

print(f"First price {events[0].price}")

#for stop_loss in [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.0325, 0.035, 0.0375, 0.04]:
#    for take_profit in [0.005, 0.01, 0.015, 0.020, 0.025, 0.030, 0.035]:
#stop_loss = 0.02
#take_profit = 0.02


start_idx = 0
episode_length = len(events)
take_profit = 0.01
valuess = []
stop_losses = []
volatilitiess = []
for stop_loss in [00.01]: # np.linspace(take_profit - 0.005, take_profit + 0.005, num=11): # [0.005, 0.007, 0.009, 0.011, 0.013, 0.015]:
    take_profit = stop_loss - 0.000
    episode_data = []
    position, values, drawdown, volatilities = simulate(start_idx, start_idx + episode_length, stop_loss, take_profit)
    episode_data.append([stop_loss, take_profit, position.wallet * events[-1].price, drawdown])
    valuess.append(values)
    stop_losses.append(stop_loss)
    volatilitiess.append(volatilities)

valuess = np.array(valuess)
stop_losses = np.array(stop_losses)
volatilitiess = np.array(volatilitiess)


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

print(valuess)
times = np.zeros(len(events))
prices = np.zeros(len(events))
for i in range(len(events)):
    times[i] = events[i].timestamp
    prices[i] = events[i].price


ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)


ax1.plot(times, prices, color='black')

for idx, values in enumerate(valuess):
    label = f"SL: {stop_losses[idx]:.2}"
    ax1.plot(times, values, label=label)
    label = f"V: {stop_losses[idx]:.2}"
    ax2.plot(times, volatilitiess[idx], label=label)

legend1 = ax1.legend()
legend2 = ax2.legend()

#ax1.ylim(0, 30000)
#ax1.ylim(0, 30000)
plt.show()

quit()


print(valuess.shape)
print(stop_losses.shape)

quit()

episode_length = 3000
step_size = 100
for start_idx in range(0, len(events), step_size):

    episode_data = []
    #valuess = []
    for stop_loss in [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.0325, 0.035, 0.0375, 0.04]:
        for take_profit in [0.005, 0.01, 0.015, 0.020, 0.025, 0.030, 0.035]:
            position, values, drawdown = simulate(start_idx, start_idx + episode_length, stop_loss, take_profit)
            #valuess.append(values)

            episode_data.append([stop_loss, take_profit, position.wallet * events[-1].price, drawdown])
            #print(f"--- stop_loss({stop_loss}) min_profit({take_profit})")
            #print(f"value({position.wallet * events[-1].price:.2f}) wallet({position.wallet:.2f}) price({events[-1].price:.2f})")

    episode_data = np.array(episode_data)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # data = np.array([[0.01, 0.005, 7882.493365287382], [0.01, 0.01, 5940.668605521221], [0.01, 0.015, 4355.852905247191], [0.01, 0.02, 9418.87807494413], [0.01, 0.025, 5457.269446459899], [0.01, 0.03, 6092.674554292229], [0.01, 0.035, 12041.669528142587], [0.02, 0.005, 6224.131543600762], [0.02, 0.01, 8293.682855665742], [0.02, 0.015, 7489.752658736529], [0.02, 0.02, 6479.014591661386], [0.02, 0.025, 8041.744161692931], [0.02, 0.03, 16852.269684544084], [0.02, 0.035, 3564.0435630970633], [0.03, 0.005, 15248.429690358638], [0.03, 0.01, 9562.51837782809], [0.03, 0.015, 15801.4449473278], [0.03, 0.02, 18382.70404789803], [0.03, 0.025, 5868.530964009983], [0.03, 0.03, 6849.081705922914], [0.03, 0.035, 6171.028387256395], [0.04, 0.005, 6603.542378502871], [0.04, 0.01, 9035.2401353807], [0.04, 0.015, 9627.561111922414], [0.04, 0.02, 7905.9845085352], [0.04, 0.025, 4544.2794939871455], [0.04, 0.03, 8886.730645571268], [0.04, 0.035, 10361.703230916513], [0.05, 0.005, 8650.668608466023], [0.05, 0.01, 6703.099442851674], [0.05, 0.015, 10428.82938395033], [0.05, 0.02, 9274.183183973755], [0.05, 0.025, 10388.112283270599], [0.05, 0.03, 9397.138075113962], [0.05, 0.035, 9322.068950603652], [0.06, 0.005, 7817.895910548697], [0.06, 0.01, 7712.366527831491], [0.06, 0.015, 8138.68274651142], [0.06, 0.02, 10734.302499940915], [0.06, 0.025, 7290.284465357713], [0.06, 0.03, 8079.190035333763], [0.06, 0.035, 8067.488631291478], [0.07, 0.005, 10233.256261507091], [0.07, 0.01, 9343.772991697897], [0.07, 0.015, 12395.651252378744], [0.07, 0.02, 8012.925159679166], [0.07, 0.025, 7156.459667041849], [0.07, 0.03, 9080.789502897067], [0.07, 0.035, 4096.47215684961]])

    X = episode_data[:, 0]
    Y = episode_data[:, 1]
    Z = (episode_data[:, 2] / events[0].price)#.clip(0.5, 4.0)


    # Plot the surface.
    surf = ax.plot_trisurf(X, Y, Z, cmap=cm.seismic, linewidth=0, antialiased=False, vmin=0.5, vmax=2.0)
    # Customize the z axis.
    ax.set_zlim(0.5, 2.0)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    #plt.show()
    fig.savefig(f"plots/fig_{start_idx:06}.png", dpi=fig.dpi * 2)
    plt.close()



    #episode_data = np.array(episode_data)
    #print(episode_data.shape)
    #quit()

"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

print(valuess)
valuess = np.array(valuess)
times = np.zeros(len(events))
prices = np.zeros(len(events))
for i in range(len(events)):
    times[i] = events[i].timestamp
    prices[i] = events[i].price


plt.plot(times, prices, color='blue')
for values in valuess:
    plt.plot(times, values)
plt.show()

quit()
"""


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

data = np.array(data)
#data = np.array([[0.01, 0.005, 7882.493365287382], [0.01, 0.01, 5940.668605521221], [0.01, 0.015, 4355.852905247191], [0.01, 0.02, 9418.87807494413], [0.01, 0.025, 5457.269446459899], [0.01, 0.03, 6092.674554292229], [0.01, 0.035, 12041.669528142587], [0.02, 0.005, 6224.131543600762], [0.02, 0.01, 8293.682855665742], [0.02, 0.015, 7489.752658736529], [0.02, 0.02, 6479.014591661386], [0.02, 0.025, 8041.744161692931], [0.02, 0.03, 16852.269684544084], [0.02, 0.035, 3564.0435630970633], [0.03, 0.005, 15248.429690358638], [0.03, 0.01, 9562.51837782809], [0.03, 0.015, 15801.4449473278], [0.03, 0.02, 18382.70404789803], [0.03, 0.025, 5868.530964009983], [0.03, 0.03, 6849.081705922914], [0.03, 0.035, 6171.028387256395], [0.04, 0.005, 6603.542378502871], [0.04, 0.01, 9035.2401353807], [0.04, 0.015, 9627.561111922414], [0.04, 0.02, 7905.9845085352], [0.04, 0.025, 4544.2794939871455], [0.04, 0.03, 8886.730645571268], [0.04, 0.035, 10361.703230916513], [0.05, 0.005, 8650.668608466023], [0.05, 0.01, 6703.099442851674], [0.05, 0.015, 10428.82938395033], [0.05, 0.02, 9274.183183973755], [0.05, 0.025, 10388.112283270599], [0.05, 0.03, 9397.138075113962], [0.05, 0.035, 9322.068950603652], [0.06, 0.005, 7817.895910548697], [0.06, 0.01, 7712.366527831491], [0.06, 0.015, 8138.68274651142], [0.06, 0.02, 10734.302499940915], [0.06, 0.025, 7290.284465357713], [0.06, 0.03, 8079.190035333763], [0.06, 0.035, 8067.488631291478], [0.07, 0.005, 10233.256261507091], [0.07, 0.01, 9343.772991697897], [0.07, 0.015, 12395.651252378744], [0.07, 0.02, 8012.925159679166], [0.07, 0.025, 7156.459667041849], [0.07, 0.03, 9080.789502897067], [0.07, 0.035, 4096.47215684961]])

X = data[:, 0]
Y = data[:, 1]
Z = data[:, 2] / events[0].price

# Plot the surface.
surf = ax.plot_trisurf (X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(0.5, 1.5)
#ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
#ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

