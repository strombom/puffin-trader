import pickle

import numpy as np
from matplotlib import pyplot as plt

from BinanceSimulator.binance_simulator import BinanceSimulator


with open(f"cache/indicators.pickle", 'rb') as f:
    data = pickle.load(f)
    # print(data)

for pair_idx, pair in enumerate(data['pairs']):
    prices = data['prices'][pair_idx]
    print(pair, min(prices), max(prices))

simulator = BinanceSimulator(initial_usdt, initial_btc, max_leverage, mark_price, initial_leverage)







fig, axs = plt.subplots(2, 1, sharex='all', gridspec_kw={'wspace': 0, 'hspace': 0})

for idx in range(data['prices'].shape[0]):
    prices = data['prices'][idx]
    axs[0].plot(prices / max(prices), label=f"{data['pairs'][idx]}")

    max_prices = np.amax(data['indicators'][idx], axis=0)
    axs[1].plot(max_prices, label=f"{data['pairs'][idx]}")

# axs[0].legend()
axs[1].legend()
plt.tight_layout()
plt.show()

