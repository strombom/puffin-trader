
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from fastai.learner import load_learner
from fastai.tabular.all import TabularPandas

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

    symbol = 'FTMUSDT'

    with open(f"cache/indicators/{symbol}.pickle", 'rb') as f:
        indicators = pickle.load(f)

    prices = []

    skip_start = 60 * 24 * 30
    skip_end = 60 * 24 * 30

    take_profit = 1.05
    stop_loss = 0.95
    steps = np.array(intrinsic_events[symbol]['steps'])
    reference_profits = mark_steps(steps=steps, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol)

    indices = np.array(intrinsic_events[symbol]['timestamps'])
    indices_end = indices.shape[0] - 1
    while indices[indices_end] >= indicators['indicators'].shape[1]:
        indices_end -= 1
    indices = indices[:indices_end]

    indicator_data = indicators['indicators'].transpose()[indices, :]  # (23321, 7)

    print("indices", indices[-1])
    #[indices, :]
    columns = list(map(str, indicators['lengths']))
    df = pd.DataFrame(data=indicator_data, columns=columns)
    """
    df = pd.DataFrame(data={
        '5': [-0.004091],
        '10': [-0.009875],
        '20': [-0.004591],
        '50': [-0.003196],
        '100': [0.000445],
        '200': [-0.000740],
        '500': [-0.000187]
    })"""

    symbols = [symbol] * df.shape[0]
    df['symbol'] = symbols

    test_dl = profit_model.dls.test_dl(df)
    predictions = profit_model.get_preds(dl=test_dl)[0][:, 2].numpy()

    fig, axs = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    axs[0].plot(steps)
    axs[0].set_yscale('log')
    axs[1].plot(reference_profits  - 0.5)
    axs[1].plot(predictions - 0.5)
    plt.tight_layout()
    plt.show()

    print()


if __name__ == '__main__':
    main()
