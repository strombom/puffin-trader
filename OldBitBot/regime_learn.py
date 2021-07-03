
import tsai
import pandas as pd
from tsai.all import *
print('tsai       :', tsai.__version__)
print('fastai     :', fastai.__version__)
print('fastcore   :', fastcore.__version__)
print('torch      :', torch.__version__)


if __name__ == '__main__':
    delta = 0.004
    n_degree = 5
    #market_states = pd.read_csv('../tmp/market_states.csv')
    #lengths = pd.read_csv('../tmp/market_states_lengths.csv')
    #print(market_states)
    #print(lengths)

    print(regression_list)
    X, y, splits = get_regression_data('AppliancesEnergy', split_data=False)
    print(X, y, splits)
    print(check_data(X, y, splits))

