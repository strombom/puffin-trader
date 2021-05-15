import pickle
from matplotlib import pyplot as plt

from BinanceSimulator.binance_simulator import BinanceSimulator


def main():

    with open(f"cache/indicators_mini.pickle", 'rb') as f:
        data = pickle.load(f)
        print(data)

    indicator_10_threshold = -0.0015

    #plt_dat = data['indicators'][12][10]
    #plt.hist(plt_dat, bins=150)
    #plt.show()

    simulator = BinanceSimulator(initial_usdt=1000, symbols=data['symbols'])

    for kline_idx in range(data['prices'].shape[1]):
        for symbol_idx, symbol in enumerate(data['symbols']):
            simulator.set_mark_price(symbol=symbol, mark_price=data['prices'][symbol_idx][kline_idx])

        symbols_of_interest = []
        for symbol_idx, symbol in enumerate(data['symbols']):
            val = data['indicators'][symbol_idx, 10, kline_idx]
            if val < indicator_10_threshold:
                symbols_of_interest.append((symbol_idx, val))
        print(symbols_of_interest)
        if len(symbols_of_interest) > 0:
            print("ok")


if __name__ == "__main__":
    main()
