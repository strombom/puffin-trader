
import numpy as np
import pandas as pd
from regime_plotter import Plotter


if __name__ == '__main__':

    lengths = pd.read_csv('../tmp/regime_data_lengths.csv')
    regime_data = pd.read_csv('../tmp/regime_data.csv')

    prediction_len = 10
    n_degrees = (regime_data.shape[1] - 5) // (2 * lengths.shape[0])
    n_samples = regime_data.shape[0]
    n_channels = lengths.shape[0]

    print(n_channels)

    spectrum_data = regime_data.to_numpy()[:, 5:].reshape((n_samples, n_degrees, n_channels, 2))
    runner_prices = regime_data['price'].to_numpy()

    plotter = Plotter(n_spectrums=n_degrees, n_samples=n_samples, n_channels=n_channels, prediction_len=prediction_len)
    plotter.plot_prices(runner_prices)

    for degree in range(n_degrees):
        plotter.spectrums[degree]['volatility']['spectrum'][:] = spectrum_data[:, degree, :, 0] * 100
        plotter.spectrums[degree]['direction']['spectrum'][:] = spectrum_data[:, degree, :, 1] * 50 + 128

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



