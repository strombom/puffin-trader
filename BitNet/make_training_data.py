
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
    with open(f"cache/intrinsic_events.pickle", 'rb') as f:
        intrinsic_events = pickle.load(f)

    intrinsic_events = dict(list(intrinsic_events.items())[:int(len(intrinsic_events) * 1.0)])

    limits = [
        (1.03, 0.93), (1.04, 0.94), (1.05, 0.95), (1.06, 0.96), (1.07, 0.97)
    ]

    lengths = pd.read_csv('cache/regime_data_lengths.csv')['length'].to_numpy()
    degrees = [1, 2, 3]

    skip_start = 60 * 24 * 30

    total_data_length = 0
    indicators = {}
    ground_truth = {}
    for symbol in intrinsic_events:
        path = f"cache/indicators"
        with open(f"{path}/{symbol}.pickle", 'rb') as f:
            indicators[symbol] = pickle.load(f)

        data_length = indicators[symbol]['indicators'].shape[2]

        ground_truth[symbol] = np.empty((len(limits), data_length))

        for limit_idx, (take_profit, stop_loss) in enumerate(limits):
            steps = np.array(intrinsic_events[symbol]['steps'])
            profits = mark_steps(steps=steps, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol)

            direction_idx = 0
            for indicator_idx in range(data_length):
                while intrinsic_events[symbol]['timestamps'][direction_idx] < indicator_idx and direction_idx + 1 < profits.shape[0]:
                    direction_idx += 1
                ground_truth[symbol][limit_idx, indicator_idx] = profits[direction_idx]

            #fig, axs = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]})
            #axs[0].plot(steps)
            #axs[0].set_yscale('log')
            #axs[1].plot(profits)
            #plt.show()

        end_of_truth = ground_truth[symbol].shape[1] - 1
        while not np.all(ground_truth[symbol][:, end_of_truth]):
            end_of_truth -= 1
        skip_end = end_of_truth + 1

        ground_truth[symbol] = ground_truth[symbol][:, skip_start:skip_end]

        total_data_length += ground_truth[symbol].shape[1]
        #break

    input_symbols = []
    data_input = np.empty((lengths.shape[0] * len(degrees), total_data_length))
    data_output = np.empty((len(limits), total_data_length))
    data_offset = 0
    for symbol in indicators:
        data_length = ground_truth[symbol].shape[1]
        if data_length <= 0:
            print("Bad length", symbol, data_length)
            continue
        tmp_indicators = indicators[symbol]['indicators'][:, :, skip_start:skip_start + data_length]
        tmp_indicators = np.transpose(tmp_indicators, (1, 0, 2))
        tmp_indicators = tmp_indicators.reshape((tmp_indicators.shape[0] * tmp_indicators.shape[1], tmp_indicators.shape[2]))
        data_input[:, data_offset:data_offset + data_length] = tmp_indicators
        data_output[:, data_offset:data_offset + data_length] = ground_truth[symbol]
        data_offset += data_length
        input_symbols.extend([symbol] * data_length)

    with open(f"cache/training_data.pickle", 'wb') as f:
        pickle.dump({
            'limits': limits,
            'skip_start': skip_start,
            'input': data_input,
            'input_symbols': input_symbols,
            'output': data_output,
            'lengths': lengths,
            'degrees': degrees
        }, f)

    print("done")


if __name__ == '__main__':
    main()
