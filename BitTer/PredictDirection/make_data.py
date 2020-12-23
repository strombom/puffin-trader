
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from IntrinsicTime.runner import Runner


if __name__ == '__main__':

    #with open(f"../cache/intrinsic_time_data.pickle", 'rb') as f:
    #    deltas, order_books, runners, runner_clock, clock_TMV, clock_R = pickle.load(f)

    #with open(f'tmp.pickle', 'wb') as f:
    #    pickle.dump((deltas, runners, runner_clock, clock_TMV, clock_R), f)

    with open(f'tmp.pickle', 'rb') as f:
        deltas, runners, runner_clock, clock_TMV, clock_R = pickle.load(f)

    #with open(f'tmp.pickle', 'rb') as f:
    #    deltas, clock_TMV, clock_R = pickle.load(f)

    def make_dataset(idx_start, idx_end):
        data_x = np.concatenate((clock_TMV[:, idx_start: idx_end], clock_R[:, idx_start: idx_end]), axis=0)
        data_y = np.where(data_x[0] > 0.0, 1, 0)

        data_x = data_x[:, :-1].transpose()
        data_y = data_y[1:]

        data_x = np.expand_dims(data_x, 0)
        data_y = np.reshape(data_y, (1, data_y.shape[0], 1))

        data_x = torch.from_numpy(data_x)
        data_y = torch.from_numpy(data_y)

        return data_x, data_y

    val_idx = clock_TMV.shape[1] * 5 // 6
    data_train = make_dataset(0, val_idx - 1)
    data_val = make_dataset(val_idx, clock_TMV.shape[1] - 1)

    print("Train", data_train[0].shape, data_train[1].shape)
    print("Val", data_val[0].shape, data_val[1].shape)

    with open(f'training_data.pickle', 'wb') as f:
        pickle.dump((data_train, data_val), f)

    quit()

    n = 100

    ax1 = plt.subplot(2, 1, 1)
    plt.scatter(runners[0].ie_times[0:n], runners[0].ie_prices[0:n], label='IE')
    plt.scatter(runners[0].os_times[0:n], runners[0].os_prices[0:n], label='OS')
    plt.plot(runners[0].os_times[0:n], runners[0].os_prices[0:n], label='OS')
    plt.legend()

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.grid(axis='y', which='both')
    plt.plot(runners[0].ie_times[0:n], clock_TMV[0][0:n], label='TMV')
    plt.scatter(runners[0].ie_times[0:n], clock_TMV[0][0:n], label='TMV')
    plt.plot(runners[0].ie_times[0:n], clock_R[0][0:n], label='R')
    plt.scatter(runners[0].ie_times[0:n], clock_R[0][0:n], label='R')
    plt.legend()

    plt.show()
