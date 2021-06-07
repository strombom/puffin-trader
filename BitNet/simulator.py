import pickle
from datetime import datetime

from fastai.learner import load_learner

from BinanceSimulator.binance_simulator import BinanceSimulator


class Portfolio:
    def __init__(self):
        pass


def main():
    with open(f"cache/filtered_symbols.pickle", 'rb') as f:
        symbols = pickle.load(f)

    indicators = {}
    for symbol in symbols:
        with open(f"cache/indicators/{symbol}.pickle", 'rb') as f:
            indicators[symbol] = pickle.load(f)

    profit_model = load_learner('model_all.pickle')

    with open(f"cache/intrinsic_events.pickle", 'rb') as f:
        intrinsic_events = pickle.load(f)

    data_length = indicators[list(indicators.keys())[0]]['prices'].shape[0]
    start_timestamp = datetime.strptime("2020-02-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    simulator = BinanceSimulator(initial_usdt=1000, symbols=symbols)

    take_profit = 1.05
    stop_loss = 0.95

    portfolio = Portfolio()

    for kline_idx in range(1, data_length):


if __name__ == '__main__':
    main()
