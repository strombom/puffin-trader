
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from indicators import SuperSmoother


def superbands(price, timesteps):
    smooth = np.copy(price)
    super_smoother = SuperSmoother(timesteps, price[0])
    for idx in range(price.shape[0]):
        smooth[idx] = super_smoother.append(price[idx])

    slope = np.zeros(smooth.shape)
    slope[1:] = smooth[1:] - smooth[:-1]

    midcomp = smooth + slope * 100

    sigma_lookback = timesteps // 6
    sqr = np.power(price - midcomp, 2)
    sigma = np.zeros(midcomp.shape)
    for idx in range(sigma_lookback, price.shape[0]):
        sigma[idx] = (np.sum(sqr[idx - sigma_lookback:idx]) / sigma_lookback) ** 0.5

    if False:
        top1 = midcomp + 0.5 * sigma
        bot1 = midcomp - 0.5 * sigma
        top2 = midcomp + 1.0 * sigma
        bot2 = midcomp - 1.0 * sigma
        top3 = midcomp + 1.5 * sigma
        bot3 = midcomp - 1.5 * sigma
        top4 = midcomp + 2.0 * sigma
        bot4 = midcomp - 2.0 * sigma

        fig, axs = plt.subplots(nrows=2, ncols=1, sharex='all', gridspec_kw={'height_ratios': [2, 1]})
        axs[0].plot(price, label="x")
        axs[0].plot(smooth, label="smooth")
        axs[0].plot(midcomp, label="midcomp")
        axs[0].plot(top1, label="top")
        axs[0].plot(bot1, label="bot")
        axs[0].plot(top2, label="top")
        axs[0].plot(bot2, label="bot")
        axs[0].plot(top3, label="top")
        axs[0].plot(bot3, label="bot")
        axs[0].plot(top4, label="top")
        axs[0].plot(bot4, label="bot")
        axs[0].legend()
        axs[0].grid(True)
        axs[1].plot(slope * 50, label="slope")
        axs[1].plot(price - midcomp, label="diff")
        axs[1].legend()
        axs[1].grid(True)
        plt.show()

    data = {
        'price': price,
        'smooth': smooth,
        'slope': slope,
        'midcomp': midcomp,
        'sigma': sigma
    }

    return data


def make_indicators(bands):
    indicators = []

    steps = [220, 100, 47, 10, 5, 2, 1]
    step_sum = sum(steps)
    print("step sum", step_sum)

    for idx in range(step_sum - 1, bands['price'].shape[0]):
        print("idx", idx)
        offset = step_sum - 1
        for step in steps:
            #print(idx - offset, idx - offset + step)
            price = np.mean(bands['price'][idx - offset:idx - offset + step])
            smooth = np.mean(bands['smooth'][idx - offset:idx - offset + step])
            slope = np.mean(bands['slope'][idx - offset:idx - offset + step])
            midcomp = np.mean(bands['midcomp'][idx - offset:idx - offset + step])
            sigma = np.mean(bands['sigma'][idx - offset:idx - offset + step])

            band_pos = (price - midcomp) / sigma
            smooth_price = (price - smooth) / price
            sigma = sigma / 20

            print(band_pos, smooth_price, slope, sigma)
            offset -= step

        if idx == 385:
            break

    return bands


if __name__ == "__main__":
    df = pd.read_csv("E:/BitCounter/test.csv")
    timestamps = df['timestamp'].to_numpy()
    prices = df['price'].to_numpy()

    bands = superbands(prices, timesteps=499)

    indicators = make_indicators(bands)


