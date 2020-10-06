
import sys
sys.path.append("../Common")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, MultipleLocator, FormatStrFormatter, AutoMinorLocator

from Common.misc import read_coastlines, string_to_timestamp
from Common.misc import calc_volatilities, calc_volatilities_regr, calc_directions


settings = {'volatility_buffer_length': 50,
            'events_filepath': '../../tmp/PD_Events/events',
            'deltas': [0.0022, 0.0033, 0.0047, 0.0068, 0.01, 0.015, 0.022],
            'data_first_timestamp': string_to_timestamp("2020-01-01 00:00:00.000"),
            'start_timestamp': string_to_timestamp("2020-01-01 00:00:00.000"),
            'end_timestamp': string_to_timestamp("2020-01-07 00:00:00.000")
            }

coastlines = read_coastlines(settings)


ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
ax1.grid(axis='y', which='both')
#ax1.yaxis.set_minor_locator(MultipleLocator(10))
#ax1.yaxis.set_major_locator(MultipleLocator(100))

#coastline = coastlines[0.0022]


for delta in coastlines:
    coastline = coastlines[delta]

    ax1.plot(coastline.timestamps_overshoot, coastline.prices_overshoot)

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
