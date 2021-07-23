
import os
import glob
import pandas as pd
from matplotlib import pyplot as plt


def main():
    path = "E:/BitBot"

    for filename in glob.glob(os.path.join(path, '*.csv')):
        print(filename)
        df = pd.read_csv(filename, sep=';')

        fix, axs = plt.subplots(3, sharex='all')
        axs[0].set_title(filename)
        axs[0].plot(df['klinecount'])
        axs[1].plot(df['delta'])
        axs[2].plot(df['price'])
        axs[2].set_yscale('log')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
