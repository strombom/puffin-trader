
import pickle
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
        deltas, order_books, runners, runner_clock, clock_TMV, clock_R = pickle.load(f)

    print(f'Deltas({len(deltas)}): {deltas}')

    for delta_idx in range(len(deltas)):
        delta = deltas[delta_idx]

        ie_os_delta_up = [] #{}
        ie_os_delta_down = [] #{}

        runner = runners[0]

        ie_idx = 1
        for os_idx in range(len(runner.os_times)):
            count = 0
            while ie_idx + 1 < len(runner.ie_times) and runner.ie_times[ie_idx] <= runner.os_times[os_idx]:
                count += 1
                ie_idx += 1

            os_price, ie_price = runner.os_prices[os_idx], runner.ie_prices[ie_idx - 1]
            delta_price = (os_price - ie_price) / ie_price / delta

            if delta_price > 0:
                #if count not in ie_os_delta_up:
                #    ie_os_delta_up[count] = []
                #if count == 2:
                #    ie_os_delta_up[1].append(delta_price + 1)
                #else:
                ie_os_delta_up.append(delta_price + count - 1)
            else:
                #if count not in ie_os_delta_down:
                #    ie_os_delta_down[count] = []
                ie_os_delta_down.append(delta_price - count + 1)

        #for count in ie_os_delta_up:
        #    print(count)
        #    ie_os_delta_up[count] = np.array(ie_os_delta_up[count])
        #for count in ie_os_delta_down:
        #    print(count)
        #    ie_os_delta_down[count] = np.array(ie_os_delta_down[count])

        print(ie_os_delta_up)
        print(ie_os_delta_down)

        plt.hist(ie_os_delta_up, bins=450, label='up')
        plt.hist(ie_os_delta_down, bins=450, label='down')

        plt.legend()
        plt.show()
        break
