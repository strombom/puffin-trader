
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers


with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
    delta, order_books, runner = pickle.load(f)

os_prices = np.array(runner.os_prices)
os_prices = np.log(os_prices)

dc_prices = np.array(runner.dc_prices)
dc_prices = np.log(dc_prices)

ie_prices = np.array(runner.ie_prices)
ie_prices = np.log(ie_prices)

"""
x = 0
for ie_idx in range(len(ie_prices)):
    print(ie_idx)
    quit()
"""





print(f'delta: {delta}')

ax1 = plt.subplot(1, 1, 1)
ax1.grid(True)
# plt.plot(times, prices, label=f'price')
# plt.plot(times, asks, label=f'ask')
# plt.plot(times, bids, label=f'bid')
# plt.plot(runner.os_times, os_prices, label=f'OS')
# plt.scatter(runner.os_times, os_prices, label=f'OS', s=5 ** 2)
# plt.scatter(runner.dc_times, dc_prices, label=f'DC', s=7 ** 2)
# plt.scatter(runner.ie_times, ie_prices, label=f'IE', s=5 ** 2)

x = np.arange(0, len(ie_prices))

plt.plot(x, ie_prices, label=f'IE')
plt.scatter(x, ie_prices, label=f'IE')

plt.legend()

plt.show()
