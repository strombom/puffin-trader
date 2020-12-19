
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    for filepath in sorted(glob.glob("log/log_*.csv"))[-20:]:

        print(f'Filepath: {filepath}')

        timestamps = []
        prices = []
        leverages = []
        values = []
        with open(filepath, 'rt') as f:
            lines = f.readlines()
            if len(lines) > 10:
                for line in lines:
                    line = line.split(',')
                    timestamps.append(float(line[0]))
                    prices.append(float(line[1]))
                    leverages.append(float(line[2]))
                    values.append(float(line[3]))
        timestamps = np.array(timestamps)
        prices = np.array(prices)
        leverages = np.array(leverages)
        #values = (np.array(values) - prices[0]) / prices[0]
        values = np.array(values)

        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312, sharex=ax1)
        ax3 = plt.subplot(313, sharex=ax1)

        ax1.plot(timestamps, prices, color='black', label=f'Price')
        ax1.legend()
        #ax1.grid(axis='y', which='both')
        #ax1.yaxis.set_minor_locator(MultipleLocator(10))
        #ax1.yaxis.set_major_locator(MultipleLocator(100))

        ax2.plot(timestamps, values, color='green', label=f'Value')
        ax2.legend()
        ax3.plot(timestamps, leverages, color='blue', label=f'Leverage')
        ax3.legend()

        #for idx in range(len(buflens)):
        #    ax2.plot(times, 10 * volatilities[idx], label=f"vol {buflens[idx]}")
        # ax2.plot(times, 10 * volatilities_regr, label="volreg")
        # ax2.plot(times, 0.05 * directions, label="dir")
        # ax2.plot(times, 0.1 * velocities, label="vel")
        #ax2.grid(axis='y', which='both')
        #legend2 = ax2.legend()

        filename = filepath.split('\\')[-1].replace('.csv', '')
        plt.savefig(f'log_png/{filename}.png')
        #plt.show()
        plt.close()

    quit()

    try:
        raise
        #with open('cache/rewards.pickle', 'rb') as f:
        #    rewards = pickle.load(f)
    except:
        rewards = []
        for filepath in sorted(glob.glob("log/log_*.csv")):
            with open(filepath, 'rt') as f:
                lines = f.readlines()
                if len(lines) < 10:
                    continue
                price_start = float(lines[0].split(',')[3])
                price_end = float(lines[-1].split(',')[3])
                reward = price_end / price_start
                rewards.append(reward)
        with open('cache/rewards.pickle', 'wb') as f:
            pickle.dump(rewards, f)

    rewards = np.array(rewards) - 1

    n = 150
    rewards = np.convolve(rewards, np.ones(n) / n, mode='valid')

    plt.plot(rewards)
    plt.show()
