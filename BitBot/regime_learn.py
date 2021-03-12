
import pandas as pd


if __name__ == '__main__':
    delta = 0.004
    n_degree = 4
    market_states = pd.read_csv('../tmp/market_states.csv')
    lengths = pd.read_csv('../tmp/market_states_lengths.csv')

    print(market_states)
    print(lengths)
    