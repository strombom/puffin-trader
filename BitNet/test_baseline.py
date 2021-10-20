import math
import pickle
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from fastai.learner import load_learner
from fastai.tabular.all import TabularPandas
from scipy.signal import lfiltic, lfilter

from cache import cache_it


@cache_it
def mark_steps(steps: np.ndarray, take_profit: float, stop_loss: float, symbol: str):
    print("mark_steps", symbol, take_profit, stop_loss)
    positions = {}
    profits = np.zeros(steps.shape[0])
    for step_idx, mark_price in enumerate(steps):
        for position_idx in list(positions.keys()):
            if mark_price >= positions[position_idx]['take_profit']:
                profits[position_idx] = 1
                del positions[position_idx]
            elif mark_price <= positions[position_idx]['stop_loss']:
                profits[position_idx] = 0
                del positions[position_idx]

        positions[step_idx] = {
            'take_profit': mark_price * take_profit,
            'stop_loss': mark_price * stop_loss
        }

    return profits


def main():
    profit_model = load_learner('model_all.pickle')

    with open(f"cache/intrinsic_events.pickle", 'rb') as f:
        intrinsic_events = pickle.load(f)

    symbols = [
        'ADAUSDT',
        'BCHUSDT',
        'BNBUSDT',
        'BTCUSDT',
        'BTTUSDT',
        'CHZUSDT',
        'DOGEUSDT',
        'EOSUSDT',
        'ETCUSDT',
        'ETHUSDT',
        'LINKUSDT',
        'LTCUSDT',
        'MATICUSDT',
        'NEOUSDT',
        'THETAUSDT',
        'TRXUSDT',
        'VETUSDT',
        'XLMUSDT',
        'XRPUSDT'
    ]
    symbol = 'ETHUSDT'

    for symbol in symbols:

        with open(f"cache/indicators/{symbol}.pickle", 'rb') as f:
            indicators = pickle.load(f)

        take_profit = 1.05
        stop_loss = 0.95
        steps = np.array(intrinsic_events[symbol]['steps'])
        reference_profits = mark_steps(steps=steps, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol)

        indices = np.array(intrinsic_events[symbol]['timestamps'])
        indices_end = indices.shape[0] - 1
        while indices[indices_end] >= indicators['indicators'].shape[2]:
            indices_end -= 1
        indices = indices[:indices_end]

        indicator_data = indicators['indicators'].transpose((1, 0, 2))
        indicator_data = indicator_data.reshape((indicator_data.shape[0] * indicator_data.shape[1], indicator_data.shape[2]))
        indicator_data = indicator_data.transpose()[indices, :]  # (29261, 27)

        degrees = [1, 2, 3]
        degree_length_inputs = []
        for degree in degrees:
            for length in indicators['lengths']:
                degree_length_inputs.append(f"{degree}-{length}")

        df = pd.DataFrame(data=indicator_data, columns=degree_length_inputs)

        for df_symbol in symbols:
            if df_symbol == symbol:
                df[df_symbol] = True
            else:
                df[df_symbol] = False

        test_dl = profit_model.dls.test_dl(df)
        predictions = profit_model.get_preds(dl=test_dl)[0][:, 2].numpy() - 0.5

        def ewma_linear_filter(array, window):
            alpha = 2 / (window + 1)
            b = [alpha]
            a = [1, alpha - 1]
            zi = lfiltic(b, a, array[0:1], [0])
            return lfilter(b, a, array, zi=zi)[0]

        center = np.zeros((predictions.shape[0], ))
        #predictions_avg5 = np.convolve(a=predictions, v=np.ones(5), mode='same') / 5
        #predictions_avg10 = np.convolve(a=predictions, v=np.ones(10), mode='same') / 10
        predictions_ema5 = ewma_linear_filter(array=predictions, window=5)
        predictions_ema50 = ewma_linear_filter(array=predictions, window=50)

        k = 0.01

        rounds = 20
        avg_val = 0
        avg_good = 0
        avg_bad = 0

        for a in range(rounds):
            eqt = 1000
            good = 0
            bad = 0
            values = []
            positions_x = []
            positions_y = []
            for idx, value in enumerate(predictions):
                if value > 0 and predictions_ema50[idx] > 0:
                    t = k * 100 ** value
                    r = random.random()
                    if r < t:
                        # print(idx, value, r, t)
                        positions_x.append(idx)
                        positions_y.append(0.0)

                        if reference_profits[idx] == 1:
                            eqt *= 1.1
                            good += 1
                        else:
                            eqt *= 0.9
                            bad += 1
                        # print("value", eqt)
            values.append(eqt)

            avg_val += good / (good + bad)
            avg_good += good
            avg_bad += bad
            #print(f"k: {k} value: {eqt} good: {good} bad: {bad} rat: {good / (good + bad)} avv {avv / (a+1)}")

        avg_val /= rounds
        avg_good /= rounds
        avg_bad /= rounds

        print(symbol, avg_val, eqt, avg_good, avg_bad)

    quit()

    ks = []
    goods = []
    bads = []

    for k in np.arange(start=0.0001, stop=0.02, step=0.0002):

        good = 0
        bad = 0
        values = []
        positions_x = []
        positions_y = []
        for idx, value in enumerate(predictions):
            if value > 0 and predictions_ema50[idx] > 0:
                t = k * 100 ** value
                r = random.random()
                if r < t:
                    #print(idx, value, r, t)
                    positions_x.append(idx)
                    positions_y.append(0.0)

                    if reference_profits[idx] == 1:
                        eqt *= 1.1
                        good += 1
                    else:
                        eqt *= 0.9
                        bad += 1
                    #print("value", eqt)
            values.append(eqt)

        print(f"k: {k} value: {eqt} good: {good} bad: {bad} rat: {good / (good + bad)}")
        ks.append(k)
        goods.append(good)
        bads.append(bad)

    goods = np.array(goods)
    bads = np.array(bads)

    plt.plot(ks, goods)
    plt.plot(ks, bads)
    plt.plot(ks, goods / (goods + bads))
    plt.show()

    quit()

    fig, axs = plt.subplots(nrows=3, sharex=True, gridspec_kw={'height_ratios': [1, 3, 1]})
    axs[0].plot(steps)
    axs[0].set_yscale('log')
    axs[1].plot(reference_profits - 0.5)
    axs[1].plot(center, label='center')
    #axs[1].plot(predictions, label='pred')
    #axs[1].plot(predictions_avg5, label='avg5')
    #axs[1].plot(predictions_avg10, label='avg10')
    axs[1].plot(predictions_ema5, label='ema')
    #axs[1].plot(predictions_ema50, label='ema')
    axs[1].scatter(x=positions_x, y = positions_y, c='r')
    axs[2].plot(values)
    axs[2].set_yscale('log')
    plt.tight_layout()
    plt.legend()
    plt.show()

    print()


if __name__ == '__main__':
    main()
