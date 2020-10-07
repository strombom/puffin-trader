
import sys
sys.path.append("../Common")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, MultipleLocator, FormatStrFormatter, AutoMinorLocator

from Common.misc import read_coastlines, string_to_timestamp
from Common.misc import calc_volatilities, calc_volatilities_regr, calc_directions
from Common.trading import Position


settings = {'volatility_buffer_length': 50,
            'events_filepath': '../../tmp/PD_Events/events',
            'deltas': [0.0015, 0.0022, 0.0033, 0.0047, 0.0068, 0.01, 0.015, 0.022],
            'data_first_timestamp': string_to_timestamp("2020-01-01 00:00:00.000"),
            'start_timestamp': string_to_timestamp("2020-01-01 00:00:00.000"),
            'end_timestamp': string_to_timestamp("2020-04-01 10:00:00.000"),
            'max_leverage': 100
            }

coastlines = read_coastlines(settings)

ax1 = plt.subplot(311)
ax2 = plt.subplot(312, sharex=ax1)
ax3 = plt.subplot(313, sharex=ax1)
#ax1.grid(axis='y', which='both')
#ax1.yaxis.set_minor_locator(MultipleLocator(10))
#ax1.yaxis.set_major_locator(MultipleLocator(100))

#coastline = coastlines[0.0022]


a = False
if a:
    for delta in [0.0033]:

        coastline = coastlines[delta]
        overshoots = abs(coastline.prices_overshoot - coastline.prices_delta)
        print(overshoots.mean())
        print(coastline.prices_overshoot.mean())
        plt.hist(overshoots, bins=50)
        plt.show()

        quit()





for delta in [0.0033]: #coastlines:

    coastline = coastlines[delta]
    position = Position(settings, mark_price=coastline.prices_overshoot[0])

    coast_length = coastline.prices_overshoot.shape[0]
    inventories = np.zeros(coast_length)
    values = np.zeros(coast_length)
    values[0] = position.get_value(coastline.prices_overshoot[0])

    asymmetry = 1.0

    for idx in range(1, coast_length):

        inventory = inventories[idx - 1]
        direction = coastline.directions[idx]

        if direction == 1:
            price = coastline.prices_delta[idx] * (1 + delta * 1.53 * asymmetry)
            while price < coastline.prices_overshoot[idx]:
                if inventory > -10:
                    inventory -= 0.5
                    order_contracts = position.calculate_order_size(leverage=inventory, mark_price=price)
                    position.limit_order(order_contracts, price)
                price = price * (1 + delta * 2.5 * asymmetry)
        elif direction == 0:
            price = coastline.prices_delta[idx] * (1 - delta * 1.53 / asymmetry)
            while price > coastline.prices_overshoot[idx]:
                if inventory < 10:
                    inventory += 0.5
                    order_contracts = position.calculate_order_size(leverage=inventory, mark_price=price)
                    position.limit_order(order_contracts, price)
                price = price * (1 - delta * 2.5 / asymmetry)

        inventory = max(-10, min(10, inventory))
        inventories[idx] = inventory

        values[idx] = position.get_value(coastline.prices_overshoot[idx])

    diff_overshoot = (coastline.prices_overshoot - coastline.prices_delta)

    ax1.plot(coastline.prices_overshoot)
    ax2.plot(values)
    ax3.plot(inventories)
    #ax2.hist(diff_overshoot, bins=300)
    break

plt.show()
quit()

#print(datetime.fromtimestamp(events[0].timestamp), '-', datetime.fromtimestamp(delta_events[-1].timestamp))









volatilities = []
volatilities_regr = []
directions = []
velocities = []

for buflen in buflens:
    settings['volatility_buffer_length'] = buflen
    volatility = calc_volatilities(events, settings)
    volatilities.append(volatility)
    volatility_regr = calc_volatilities_regr(events, settings)
    volatilities_regr.append(volatility_regr)
    direction, velocity = calc_directions(events, settings)
    directions.append(direction)
    velocities.append(velocity)


times = np.zeros(len(events))
prices = np.zeros(len(events))
for i in range(len(events)):
    times[i] = events[i].timestamp
    prices[i] = events[i].price

ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)

ax1.plot(times, prices, color='black')
ax1.grid(axis='y', which='both')
ax1.yaxis.set_minor_locator(MultipleLocator(10))
ax1.yaxis.set_major_locator(MultipleLocator(100))


for idx in range(len(buflens)):
    ax2.plot(times, 10 * volatilities[idx], label=f"vol {buflens[idx]}")
#ax2.plot(times, 10 * volatilities_regr, label="volreg")
#ax2.plot(times, 0.05 * directions, label="dir")
#ax2.plot(times, 0.1 * velocities, label="vel")
ax2.grid(axis='y', which='both')
legend2 = ax2.legend()

plt.show()
