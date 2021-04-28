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

simulator = BinanceSimulator(initial_usdt=1000, pairs=data['pairs'])

previous_pair_idx = 0
max_pairs = []
values = []
for idx in range(data['prices'].shape[1]):
    max_pair_idx, max_pair_val = 0, -1.0
    for pair_idx, pair in enumerate(data['pairs']):
        val = np.amax(data['indicators'][pair_idx, :, idx])
        if val > max_pair_val:
            max_pair_val = val
            max_pair_idx = pair_idx
    # print(f'max_pair_idx: {max_pair_idx}, max_pair_val: {max_pair_val}')
    max_pairs.append(max_pair_val)
    if max_pair_idx != previous_pair_idx:
        for pair_idx, pair in enumerate(data['pairs']):
            simulator.set_mark_price(pair=pair, mark_price=data['prices'][pair_idx][idx])

        simulator.sell_pair(pair=data['pairs'][previous_pair_idx])
        order_size = simulator.calculate_order_size(leverage=1, pair=data['pairs'][max_pair_idx])
        simulator.market_order(order_size=order_size, pair=data['pairs'][max_pair_idx])

        previous_pair_idx = max_pair_idx

    values.append(simulator.get_value_usdt())

fig, axs = plt.subplots(3, 1, sharex='all', gridspec_kw={'wspace': 0, 'hspace': 0})

for idx in range(data['prices'].shape[0]):
    prices = data['prices'][idx]
    axs[0].plot(prices / max(prices), label=f"{data['pairs'][idx]}")

    max_prices = np.amax(data['indicators'][idx], axis=0)
    axs[1].plot(max_prices, label=f"{data['pairs'][idx]}")

# axs[1].plot(max_pairs, label="max_pairs")
axs[2].plot(values, label="Value UDST")

# axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.tight_layout()
plt.show()

