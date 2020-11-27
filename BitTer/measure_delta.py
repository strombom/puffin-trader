
import pickle
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
        deltas, order_books, runners, runner_clock, clock_TMV, clock_R = pickle.load(f)

    print(f'Deltas({len(deltas)}): {deltas}')

    for delta_idx in range(len(deltas)):
        delta = deltas[delta_idx]

        ie_os_delta_up = []
        ie_os_delta_down = []

        runner = runners[0]

        os_idx = 1
        ie_price = runner.ie_prices[0]

        for ie_idx in range(len(runner.ie_times)):
            if runner.ie_times[ie_idx] >= runner.os_times[os_idx]:
                delta_price = (runner.os_prices[os_idx] - ie_price) / ie_price / delta
                if delta_price > 20 or delta_price < -20:
                    print(f'hm {runner.ie_times[ie_idx]}, {runner.ie_prices[ie_idx]}, {delta_price}')

                if delta_price > 0:
                    ie_os_delta_up.append(delta_price)
                else:
                    ie_os_delta_down.append(-delta_price)
                os_idx += 1
                if os_idx == len(runner.os_times):
                    break
            ie_price = runner.ie_prices[ie_idx]

        ie_os_delta_up = np.array(ie_os_delta_up)
        ie_os_delta_down = np.array(ie_os_delta_down)

        print(ie_os_delta_up)
        print(ie_os_delta_down)

        plt.hist(ie_os_delta_up, bins=50, label='up')
        plt.hist(ie_os_delta_down, bins=50, label='down')

        plt.legend()
        plt.show()
        break
