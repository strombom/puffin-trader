
import pandas as pd
from regime_plotter import Plotter


if __name__ == '__main__':

    plotter = Plotter(5)

    plotter.show()


    quit()
    lengths = pd.read_csv('../tmp/regime_data_lengths.csv')
    regime_data = pd.read_csv('../tmp/regime_data.csv')

    regime_data['duration'] /= 100000000
    regime_data['volume'] /= 2000
    regime_data['delta'] /= 0.008



