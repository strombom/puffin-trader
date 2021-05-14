import pickle
from matplotlib import pyplot as plt


def main():

    with open(f"cache/indicators_mini.pickle", 'rb') as f:
        data = pickle.load(f)
        print(data)

    indicator_10_threshold = 0.0015

    plt_dat = data['indicators'][12][10]
    plt.hist(plt_dat, bins=150)
    plt.show()


if __name__ == "__main__":
    main()
