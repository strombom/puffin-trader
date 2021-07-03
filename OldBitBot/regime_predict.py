
import numpy as np
import pandas as pd
from tsai.all import *

from regime_plotter import Plotter


if __name__ == '__main__':

    lengths = pd.read_csv('../tmp/regime_data_lengths.csv')
    regime_data = pd.read_csv('../tmp/regime_data.csv')

    regime_data['duration'] /= 100000000
    regime_data['volume'] /= 2000
    regime_data['delta'] /= 0.008

    prediction_len = 40
    n_degrees = (regime_data.shape[1] - 5) // (2 * lengths.shape[0])
    n_samples = regime_data.shape[0]
    n_channels = lengths.shape[0]

    print(n_channels)

    spectrum_data = regime_data.to_numpy()[:, 5:].reshape((n_samples, n_degrees, n_channels, 2))
    runner_prices = regime_data['price'].to_numpy()

    regime_data_raw = regime_data.to_numpy()[:, 2:]
    regime_predictor = load_learner_all('regime_model')

    def prediction_callback(sample_idx):
        n_timesteps = 10
        if n_timesteps < sample_idx < n_samples - prediction_len:
            start, end = sample_idx - n_timesteps, sample_idx + prediction_len
            for degree in range(n_degrees):
                img = spectrum_data[start:end, degree, :, 0] * 100
                plotter.spectrums[degree]['volatility']['target'].setImage(img, levels=(0, 255))
                plotter.spectrums[degree]['direction']['target'].setImage(spectrum_data[start:end, degree, :, 1] * 50 + 128, levels=(0, 255))

            predictions = np.empty([273, n_timesteps + prediction_len])
            predictions[:, 0:n_timesteps] = regime_data_raw[sample_idx - n_timesteps:sample_idx, :].transpose()

            for prediction_idx in range(prediction_len):
                features = predictions[:, prediction_idx:prediction_idx + n_timesteps]
                prediction = regime_predictor.get_X_preds(features)[0][0]
                # print(prediction.shape)
                # print(predictions[:, n_timesteps + prediction_idx].shape)
                predictions[:, n_timesteps + prediction_idx] = prediction
            predictions = predictions[:, :].transpose()

            for degree in range(n_degrees):
                ch_idx = 3 + degree * n_channels * 2
                img = predictions[:, ch_idx:ch_idx + n_channels * 2:2] * 100
                plotter.spectrums[degree]['volatility']['prediction'].setImage(img, levels=(0, 255))
                ch_idx = 3 + degree * n_channels * 2 + 1
                plotter.spectrums[degree]['direction']['prediction'].setImage(predictions[:, ch_idx:ch_idx + n_channels * 2:2] * 50 + 128, levels=(0, 255))


    plotter = Plotter(n_spectrums=n_degrees,
                      n_samples=n_samples,
                      n_channels=n_channels,
                      prediction_len=prediction_len,
                      prediction_callback=prediction_callback)
    plotter.plot_prices(runner_prices)

    for degree in range(n_degrees):
        plotter.spectrums[degree]['volatility']['spectrum'].setImage(spectrum_data[:, degree, :, 0] * 100, levels=(0, 255))
        plotter.spectrums[degree]['direction']['spectrum'].setImage(spectrum_data[:, degree, :, 1] * 50 + 128, levels=(0, 255))

        print(np.max(spectrum_data[:, degree, :, 1]))
        print(np.min(spectrum_data[:, degree, :, 1]))
        #print(vol_spec[:].shape)
        #print(spectrum_data[:, degree, :, 0].shape)
        #quit()

    plotter.show()

    quit()
    lengths = pd.read_csv('../tmp/regime_data_lengths.csv')
    regime_data = pd.read_csv('../tmp/regime_data.csv')

    regime_data['duration'] /= 100000000
    regime_data['volume'] /= 2000
    regime_data['delta'] /= 0.008



