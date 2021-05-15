import pickle
from matplotlib import pyplot as plt

from BinanceSimulator.binance_simulator import BinanceSimulator


def main():

    with open(f"cache/indicators_mini.pickle", 'rb') as f:
        data = pickle.load(f)
        print(data)

    indicator_10_threshold = 0.0015

    #plt_dat = data['indicators'][12][10]
    #plt.hist(plt_dat, bins=150)
    #plt.show()

    simulator = BinanceSimulator(initial_usdt=1000, symbols=data['symbols'])

    for kline_idx in range(data['prices'].shape[1]):
        for symbol_idx, symbol in enumerate(data['symbols']):
            simulator.set_mark_price(symbol=symbol, mark_price=data['prices'][symbol_idx][kline_idx])



if __name__ == "__main__":
    main()
