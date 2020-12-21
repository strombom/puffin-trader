import pickle
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
        deltas, order_books, runners, runner_clock, clock_TMV, clock_R = pickle.load(f)

    print(f'Deltas({len(deltas)}): {deltas}')

    print(f"TMV {np.min(clock_TMV)} {np.max(clock_TMV)}")
    print(f"R {np.min(clock_R)} {np.max(clock_R)}")

    # same_timestamps_0 = []
    # same_timestamps_1 = []
    # same_prices_0 = []
    # same_prices_1 = []
    # for idx_clock, timestamp in enumerate(runner_clock.ie_times):
    #     tmv = clock_TMV[0][idx_clock]
    #     if -0.05 < tmv < 0.05:
    #         same_timestamps_0.append(timestamp)
    #         same_prices_0.append(runner_clock.ie_prices[idx_clock])
    #         print(timestamp, runner_clock.ie_prices[idx_clock])
    #     if 1.58 < tmv < 1.59:
    #         same_timestamps_1.append(timestamp)
    #         same_prices_1.append(runner_clock.ie_prices[idx_clock])
    #         #print(timestamp, runner_clock.ie_prices[idx_clock])




    print(clock_TMV.shape)

    #clock_TMV = np.abs(clock_TMV * np.heaviside(clock_TMV, 0))

    tmv_pos = clock_TMV[3]

    #tmv_pos = np.log1p(clock_TMV.clip(min=0)) / 5
    #tmv_neg = np.log1p(-clock_TMV.clip(max=0)) / 5

    #print("shape", tmv_pos.shape)

    #plt.subplot(2, 1, 1)
    for idx in [0]:  # range(len(deltas)):
        #plt.plot(clock_TMV[idx], label=f'TMV {idx}')
        plt.hist(clock_TMV[idx], bins=200, label=f'TMV {idx}')

    #plt.plot(clock_R[3], label=f'R {3}')
    #plt.legend()

    # #plt.subplot(2, 1, 2)
    #tmv_pos = np.log1p(clock_TMV.clip(min=0)) / 3
    #tmv_neg = np.log1p(-clock_TMV.clip(max=0)) / 3

    #plt.plot(, label=f'TMVn {3}')
    #plt.plot(clock_R[3], label=f'R {3}')

    #plt.hist(tmv_pos, bins=400)
    #for idx in range(len(deltas)):
    #    plt.hist(clock_R[idx], bins=200, label=f'R {idx}')

    #plt.hist(clock_TMV[0], bins=100)
    #plt.plot(runner_clock.ie_times, runner_clock.ie_prices)
    #plt.plot(runner_clock.os_times, runner_clock.os_prices)
    #plt.scatter(same_timestamps_0, same_prices_0, color='red')
    #plt.scatter(same_timestamps_1, same_prices_1, color='blue')

    plt.legend()
    plt.show()
