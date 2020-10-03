
import sys
import csv
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def string_to_timestamp(date):
    return datetime.timestamp(datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")) * 1000


filename = "../PD_Events/events.csv"


maker_fee = -0.00025
taker_fee = 0.00075

settings = {'max_leverage': 10.0,
            'volatility_buffer_length': 250,
            'leverage_factor': 100,
            'take_profit': 0.01,
            'stop_loss': 0.01,
            'data_first_timestamp': string_to_timestamp("2020-01-01 00:00:00.000"),
            'start_timestamp': string_to_timestamp("2020-01-01 00:00:00.000"),
            'max_order_value': 10.0,
            'min_leverage_take_profit': 0.1,
            'min_leverage_stop_loss': 0.1
            }

class Event:
    def __init__(self, timestamp, price):
        self.timestamp = (settings['data_first_timestamp'] + int(timestamp)) / 1000
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

        if settings['data_first_timestamp'] + ts > settings['start_timestamp']:
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
        self.contracts = 0 #initial_leverage * self.wallet * self.price
        self.take_profit = take_profit
        self.min_profit = min_profit
        self._update()

    def _update(self):
        if self.contracts == 0:
            self.direction = 1
        else:
            self.direction = self.contracts / abs(self.contracts)
        self.take_profit_price = self.price * (1 + self.direction * take_profit)
        self.stop_loss_price = self.price * (1 - self.direction * stop_loss)

        if self.wallet < 0.2:
            self.wallet = 0
            self.contracts = 0
            return

    def get_leverage(self, mark_price):
        if self.wallet == 0:
            return 0
        value = self.contracts / mark_price
        leverage = value / self.wallet
        return leverage

    def get_value(self, mark_price):
        if self.wallet == 0:
            return 0
        upnl = self.contracts * (1 / self.price - 1 / mark_price)
        return (self.wallet + upnl) * mark_price

    def market_order(self, order_leverage, mark_price, timestamp):

        # print('market_order', wallet, pos_price, pos_contracts, order_contracts, mark_price)
        if self.wallet == 0:
            return

        upnl = self.contracts * (1 / self.price - 1 / mark_price)

        #if timestamp > 1578035300:
        #    print("a")
        #elif timestamp > 1578416400:
        #    print("a")

        max_contracts = settings['max_leverage'] * (self.wallet + upnl) * mark_price
        max_contracts = min(max_contracts, settings['max_order_value'] * mark_price)

        margin = self.wallet * min(max(order_leverage, -settings['max_leverage']), settings['max_leverage'])
        order_contracts = min(max(margin * mark_price, -max_contracts), max_contracts) - self.contracts

        # Fee
        fee = taker_fee * abs(order_contracts) / mark_price
        self.wallet -= fee

        # Realised profit and loss
        # Wallet only changes when abs(contracts) decrease
        realised = self.wallet
        if (self.contracts > 0) and (order_contracts < 0):
            self.wallet += (1 / self.price - 1 / mark_price) * min(-order_contracts, self.contracts)
        elif (self.contracts < 0) and (order_contracts > 0):
            self.wallet += (1 / self.price - 1 / mark_price) * max(-order_contracts, self.contracts)

        realised = self.wallet - realised
        #if realised != 0:
        #    print("realise", self.wallet, realised, fee, realised - fee)

        # Calculate average entry price
        if (self.contracts >= 0 and order_contracts > 0) or (self.contracts <= 0 and order_contracts < 0):
            self.price = (self.contracts * self.price + order_contracts * mark_price) / (self.contracts + order_contracts)
        elif (self.contracts >= 0 and self.contracts + order_contracts < 0) or (self.contracts <= 0 and self.contracts + order_contracts > 0):
            self.price = mark_price

        #print(f"t({datetime.fromtimestamp(timestamp)}) price({mark_price}) contr({order_contracts})")

        # Calculate position contracts
        self.contracts += order_contracts
        self._update()


def simulate(start_idx, end_idx, stop_loss, take_profit, min_leverage_stop_loss=3.0, min_leverage_take_profit=2.0):
    volatility = 0.0
    wallets = []
    volatilities = []
    leverages = []
    vola_prices = []
    values = []
    drawdown = 0.0
    peak_price = events[0].price
    position = Position(events[0].price, settings['max_leverage'], stop_loss, take_profit)
    for idx in range(start_idx, end_idx):
        event = events[idx]
        leverage = volatility * settings['leverage_factor']

        if leverage > settings['min_leverage_stop_loss'] and position.direction > 0 and event.price < position.stop_loss_price:
            leverage = -leverage #-3
            mark_price = event.price #min(events[idx - 1].price, position.stop_loss_price)
            position.market_order(leverage, mark_price, event.timestamp)

        elif leverage > settings['min_leverage_stop_loss'] and position.direction < 0 and event.price > position.stop_loss_price:
            leverage = leverage
            mark_price = event.price #max(events[idx - 1].price, position.stop_loss_price)
            position.market_order(leverage, mark_price, event.timestamp)

        elif leverage > settings['min_leverage_take_profit'] and position.direction > 0 and event.price > position.take_profit_price:
            leverage = -leverage #-3
            position.market_order(leverage, event.price, event.timestamp)

        elif leverage > settings['min_leverage_take_profit'] and position.direction < 0 and event.price < position.take_profit_price:
            leverage = leverage
            position.market_order(leverage, event.price, event.timestamp)


        peak_price = max(peak_price, event.price)
        if event.price < peak_price:
            drawdown = max(drawdown, (peak_price - event.price) / peak_price)
        values.append(position.wallet * event.price) #  position.get_value(event.price))

        vola_prices.append(event.price)
        if len(vola_prices) > settings['volatility_buffer_length']:
            vola_prices = vola_prices[1:]
        volatility = max(vola_prices) / min(vola_prices) - 1
        volatilities.append(volatility)

        leverages.append(position.get_leverage(event.price))
        wallets.append(position.wallet)

        # print(f"go long, value({position.wallet * event.price} price({event.price})")
    return position, values, drawdown, volatilities, leverages, wallets

#print(f"First price {events[0].price}")

#for stop_loss in [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.0325, 0.035, 0.0375, 0.04]:
#    for take_profit in [0.005, 0.01, 0.015, 0.020, 0.025, 0.030, 0.035]:
#stop_loss = 0.02
#take_profit = 0.02


start_idx = 0
episode_length = len(events)
valuess = []
stop_losses = []
volatilitiess = []
leveragess = []
walletss = []
take_profit = settings['take_profit']
stop_loss = settings['stop_loss']
min_leverage_stop_loss = 4.5
min_leverage_take_profit = 1.5

# np.linspace(take_profit - 0.005, take_profit + 0.005, num=11): # [0.005, 0.007, 0.009, 0.011, 0.013, 0.015]:
#for vbl in [200, 210, 220, 230, 240, 250, 260]:
#for stop_loss in np.linspace(0.001, 0.004, 7):
for a in [1]:
    #take_profit = stop_loss

    episode_data = []
    position, values, drawdown, volatilities, leverages, wallets = simulate(start_idx, start_idx + episode_length, stop_loss, take_profit, min_leverage_stop_loss, min_leverage_take_profit)
    episode_data.append([stop_loss, take_profit, position.wallet * events[-1].price, drawdown])
    valuess.append(values)
    stop_losses.append(stop_loss)
    volatilitiess.append(volatilities)
    leveragess.append(leverages)
    walletss.append(wallets)

valuess = np.array(valuess)
stop_losses = np.array(stop_losses)
volatilitiess = np.array(volatilitiess)
walletss = np.array(walletss)


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


ax1 = plt.subplot(311)
ax2 = plt.subplot(312, sharex=ax1) # Volatility
ax3 = plt.subplot(313, sharex=ax1) # Leverage


ax1.plot(times, prices, color='black')

for idx, values in enumerate(valuess):
    label = f"SL: {stop_losses[idx]}"
    #ax2.plot(times, values, label=label)
    #ax2.set_yscale('log')

    ax2.plot(times, walletss[idx], label=label)
    ax2.set_yscale('log')

    #label = f"V: {stop_losses[idx]}"
    #ax2.plot(times, volatilitiess[idx], label=label)
    label = f"SL: {stop_losses[idx]}"
    ax3.plot(times, leveragess[idx], label=label)

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

