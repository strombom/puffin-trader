
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


if __name__ == '__main__':
    with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
        deltas, order_books, runners, runner_clock, clock_TMV, clock_R = pickle.load(f)

    print(f'Deltas({len(deltas)}): {deltas}')

    #x = np.arange(len(runner_clock.ie_times))

    prices = np.empty(len(order_books))
    asks = np.empty(len(order_books))
    bids = np.empty(len(order_books))
    times = np.empty(len(order_books))
    for idx, order_book in enumerate(order_books):
        prices[idx] = order_book.mid
        asks[idx] = order_book.ask
        bids[idx] = order_book.bid
        times[idx] = datetime.timestamp(order_book.timestamp)

    plt.plot(times, prices, label=f'price')
    plt.plot(times, asks, label=f'ask')
    plt.plot(times, bids, label=f'bid')

    #plt.plot(runner_clock.ie_times, runner_clock.ie_prices, label='IE C')
    #plt.scatter(runner_clock.ie_times, runner_clock.ie_prices, label='IE C')
    #plt.plot(runner_clock.ie_times, runner_clock.ie_prices_max)
    #plt.plot(runner_clock.ie_times, runner_clock.ie_prices_min)
    #plt.plot(runners[0].os_times, runners[0].os_prices, label='OS 0')
    #plt.scatter(runners[0].ie_times, runners[0].ie_prices, label='IE 0')
    #plt.plot(runners[1].os_times, runners[1].os_prices, label='OS 1')
    for i in [0]:
        plt.plot(runners[i].os_times, runners[i].os_prices, label=f'OS {i}')
        plt.scatter(runners[i].ie_times, runners[i].ie_prices, label=f'IE {i}')
        plt.scatter(runners[i].os_times, runners[i].os_prices, label=f'OS {i}')
        plt.scatter(runners[i].dc_times, runners[i].dc_prices, label=f'DC {i}')

    #plt.plot()

    plt.legend()
    plt.show()
