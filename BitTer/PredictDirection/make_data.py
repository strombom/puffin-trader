
import pickle
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

    print(len(deltas), deltas)
    print(clock_TMV.shape)

    #(60, 22641)
    n_history = 50

    print(clock_TMV[0][0:10])
    print(runners[0].ie_prices[0:10])

    n = 100

    ax1 = plt.subplot(2, 1, 1)
    plt.scatter(runners[0].ie_times[0:n], runners[0].ie_prices[0:n], label='IE')
    plt.scatter(runners[0].os_times[0:n], runners[0].os_prices[0:n], label='OS')
    plt.plot(runners[0].os_times[0:n], runners[0].os_prices[0:n], label='OS')
    plt.legend()

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(runners[0].ie_times[0:n], clock_TMV[0][0:n], label='TMV')
    plt.scatter(runners[0].ie_times[0:n], clock_TMV[0][0:n], label='TMV')
    plt.legend()

    plt.show()

    #for idx in range(clock_TMV.shape[1] - n_history):




