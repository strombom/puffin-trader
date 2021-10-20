import pickle
import numpy as np
from matplotlib import pyplot as plt

from prediction_filter import PredictionFilter


def main():

    file_path = f"cache/predictions.pickle"
    try:
        with open(file_path, 'rb') as f:
            prediction_data = pickle.load(f)
            symbols = prediction_data['symbols']
            predictions = prediction_data['predictions']
    except FileNotFoundError:
        print("Load predictions fail ", file_path)
        quit()

    symbol = 'BTCUSDT'
    a = []
    for prediction in predictions:
        if symbol in prediction:
            a.append(prediction[symbol]) # * 2 - 1)
        else:
            pass
            #a.append(0.0)

    a = np.array(a)

    for i in [0, 0]:
        plt.plot(a[:, i], label=[1.020, 1.025, 1.030, 1.035, 1.040, 1.045, 1.050, 1.055, 1.060][i])
    #fig, axs = plt.subplots(7, sharex='all')
    #for i in range(5):
    #    axs[i].plot(a[:, i])
    #    axs[i].set_ylim(0, 0.5)
    #fig.tight_layout()
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
    plt.show()
    quit()

    symbol = 'DOGEUSDT'
    a = []
    for prediction in predictions:
        if symbol in prediction:
            a.append(prediction[symbol])
        else:
            a.append(0.0)

    predictions = np.array(a)

    #with open('tmp_prediction_filter.pickle', 'wb') as f:
    #    pickle.dump(a, f)

    #with open('tmp_prediction_filter.pickle', 'rb') as f:
    #    predictions = pickle.load(f)

    pf = PredictionFilter()

    for value in predictions:
        pf.append(value)

    num = 0
    for idx in range(predictions.shape[0]):
        if predictions[idx] > pf.threshold[idx]:
            num += 1

    print(num, predictions.shape[0])

    plt.plot(predictions)
    plt.plot(pf.threshold)
    plt.plot(pf.smooth)
    plt.show()


if __name__ == '__main__':
    main()
