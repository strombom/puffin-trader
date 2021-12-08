
#import scipy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

from bybit_simulator import ByBitSimulator
from indicators import SuperSmoother


def simulate(ind_top, ind_mid, ind_bot):
    simulator = ByBitSimulator(initial_usdt=10000, symbols=['BTCUSDT'])

    pass


def polysmooth(x, timesteps):
    y = np.copy(x)
    for idx in range(winlen, y.shape[0]):
        start, end = idx - winlen, idx
        xp = np.arange(start, end)
        yp = np.poly1d(np.polyfit(xp, x[start:end], 2))
        y[idx] = yp(idx)
    return y


def supersmooth(x, timesteps):
    y = np.copy(x)
    super_smoother = SuperSmoother(winlen, x[0])
    for idx in range(y.shape[0]):
        y[idx] = super_smoother.append(x[idx])
    return y


def superbands(x, timesteps):
    mid = np.copy(x)
    super_smoother = SuperSmoother(timesteps, x[0])
    for idx in range(x.shape[0]):
        mid[idx] = super_smoother.append(x[idx])

    slope = np.zeros(mid.shape)
    slope[1:] = mid[1:] - mid[:-1]

    midcomp = mid + slope * 100

    sigma_lookback = timesteps // 6
    sqr = np.power(x - midcomp, 2)
    sigma = np.zeros(midcomp.shape)
    for idx in range(sigma_lookback, x.shape[0]):
        sigma[idx] = (np.sum(sqr[idx - sigma_lookback:idx]) / sigma_lookback) ** 0.5

    top1 = midcomp + 0.5 * sigma
    bot1 = midcomp - 0.5 * sigma
    top2 = midcomp + 1.0 * sigma
    bot2 = midcomp - 1.0 * sigma
    top3 = midcomp + 1.5 * sigma
    bot3 = midcomp - 1.5 * sigma
    top4 = midcomp + 2.0 * sigma
    bot4 = midcomp - 2.0 * sigma

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex='all', gridspec_kw={'height_ratios': [2, 1]})
    axs[0].plot(x, label="x")
    axs[0].plot(mid, label="mid")
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
    axs[1].plot(x - midcomp, label="diff")
    axs[1].legend()
    axs[1].grid(True)
    plt.show()

    sqr = np.power(x - mid, 2)
    top = np.copy(mid)
    for idx in range(timesteps, x.shape[0]):
        sigma = (np.sum(sqr[idx - timesteps:idx]) / timesteps) ** 0.5

        """

        """

        a = 1
        #top[idx]

    return mid


if __name__ == "__main__":
    df = pd.read_csv("E:/BitCounter/test.csv")
    timestamps = df['timestamp'].to_numpy()
    prices = df['price'].to_numpy()

    if True:
        smooths = []
        for wl, po in ((499, 2), ):
            #smooth = scipy.signal.savgol_filter(prices, window_length=wl, polyorder=po)
            #smooths.append((smooth, f"Savgol {wl} {po}"))

            #smooth = polysmooth(prices, winlen=wl)
            #smooths.append((smooth, f"Polysmooth {wl}"))

            smooth = superbands(prices, timesteps=wl)
            smooths.append((smooth, f"Supersmooth {wl}"))
        with open('cache/smooths.pickle', 'wb') as f:
            pickle.dump(smooths, f, pickle.HIGHEST_PROTOCOL)

    else:
        with open('cache/smooths.pickle', 'rb') as f:
            smooths = pickle.load(f)

    x_buy, y_buy, t_buy = [], [], []
    x_sell, y_sell, t_sell = [], [], []
    direction = -1
    for idx in range(500, smooths[0][0].shape[0]):
        fast, slow = smooths[0][0][idx], smooths[1][0][idx]
        if direction == 1 and fast < slow:
            direction = -1
            x_sell.append(idx)
            y_sell.append(prices[idx])
            t_sell.append(timestamps[idx])
        elif direction == -1 and fast > slow:
            direction = 1
            x_buy.append(idx)
            y_buy.append(prices[idx])
            t_buy.append(timestamps[idx])

    if False:
        dfs = []
        for i in range(25, 31):
            dfs.append(pd.read_csv(f'E:/BitCounter/csv/BTCUSDT2020-03-{i}.csv')[::-1])
        df = pd.concat(dfs).reset_index(drop=True)

        simulator = ByBitSimulator(initial_usdt=10000, symbols=['BTCUSDT'])

        values = []
        timestamps = []

        buy_idx = 0
        sell_idx = 0
        for idx in range(500, df.shape[0]):
            timestamp = df['timestamp'][idx]
            if timestamp > t_buy[buy_idx] + 0.2:
                price = df['price'][idx]
                simulator.mark_price['BTCUSDT'] = price
                cash = simulator.get_cash_usdt()
                simulator.market_order(cash / price * 0.99, symbol='BTCUSDT')
                #simulator.market_order(-simulator.wallet['BTCUSDT'], symbol='BTCUSDT')

                equity = simulator.get_equity_usdt()
                values.append(equity)
                timestamps.append(timestamp)
                print("Buy", equity)
                buy_idx += 1
                if buy_idx == len(t_buy):
                    break
            if timestamp > t_sell[sell_idx] + 0.2:
                price = df['price'][idx]
                simulator.mark_price['BTCUSDT'] = price
                #cash = simulator.get_cash_usdt()
                #simulator.market_order(cash / price * 0.99, symbol='BTCUSDT')
                simulator.market_order(-simulator.wallet['BTCUSDT'], symbol='BTCUSDT')
                equity = simulator.get_equity_usdt()
                values.append(equity)
                timestamps.append(timestamp)
                print("Sell", equity)
                sell_idx += 1
                if sell_idx == len(t_sell):
                    break

        plt.plot(timestamps, values)
        plt.show()
        quit()

    #data_length = prices.shape[0]
    #smooths.append((prices, "Price"))

    """
    for time_constant in [
            #10, 16,
            #25,
            #40,
            #63,
            100,
            #160,
            #250,
            #400,
            #630,
            #1000,
            #1600,
            #2500,
            # 4000,
            #6300,
            #10000, 20000
        ]:
        super_smoother = SuperSmoother(time_constant, prices.iloc[0])
        #smooth = np.zeros(data_length)
        #derivatives = np.zeros(data_length)
        #prev_out = prices.iloc[0]

        #for idx, price in prices.items():
        #    out = super_smoother.append(price)
        #    if abs(out - price) > 80:
        #        super_smoother = SuperSmoother(time_constant, price)
        #        out = price
        #        prev_out = out
        #    smooth[idx] = out
        #    derivatives[idx] = (out / prev_out - 1) * 450_000
        #    prev_out = out

        #smooths.append((smooth, "Smooth " + str(time_constant)))
        #smooths.append((smooth + derivatives, "Smoothx " + str(time_constant)))
        #smooths.append((smooth + 10 + derivatives, "Smooth+ " + str(time_constant)))
        #smooths.append((smooth - 10 + derivatives, "Smooth- " + str(time_constant)))
    """

    print(smooths)

    #simulate(ind_top=smooths[1][0], ind_mid=smooths[0][0], ind_bot=smooths[2][0])

    #diff = df['price'] - smooths[0][0] #smooths[1][0]
    #print(diff)

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex='all', gridspec_kw={'height_ratios': [2, 1]})
    axs[0].plot(prices, label="Price")
    for smooth in smooths:
        axs[0].plot(smooth[0], label=smooth[1])
    #ax.plot(df['timestamp'], df['price'])

    axs[0].scatter(x_buy, y_buy, marker=(5, 1), s=100, c='green')
    axs[0].scatter(x_sell, y_sell, marker=(5, 1), s=100, c='red')

    axs[0].legend()
    axs[0].grid(True)
    #axs[1].plot(np.abs(diff))
    #axs[1].plot(derivatives)
    axs[1].grid(True)
    plt.show()


