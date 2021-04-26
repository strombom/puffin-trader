import pickle

with open(f"cache/indicators.pickle", 'rb') as f:
    data = pickle.load(f)
    # print(data)

for pair_idx, pair in enumerate(data['pairs']):
    prices = data['prices'][pair_idx]
    print(pair, min(prices), max(prices))

for idx in range(data['prices'].shape[1]):
    print(idx)
