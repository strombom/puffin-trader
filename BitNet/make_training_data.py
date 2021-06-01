
import pickle
import numpy as np
from matplotlib import pyplot as plt

from cache import cache_it


@cache_it
def mark_steps(steps: np.ndarray, take_profit: float, stop_loss: float):
    print("mark_steps", take_profit, stop_loss)
    positions = {}
    profits = np.zeros(steps.shape[0])
    for step_idx, mark_price in enumerate(steps):
        for position_idx in list(positions.keys()):
            if mark_price >= positions[position_idx]['take_profit']:
                profits[position_idx] = 1
                del positions[position_idx]
            elif mark_price <= positions[position_idx]['stop_loss']:
                profits[position_idx] = -1
                del positions[position_idx]

        positions[step_idx] = {
            'take_profit': mark_price * take_profit,
            'stop_loss': mark_price * stop_loss
        }

    return profits


def main():
    with open(f"cache/intrinsic_events.pickle", 'rb') as f:
        intrinsic_events = pickle.load(f)

    limits = [
        (1.05, 0.85), (1.05, 0.90), (1.05, 0.95),
        (1.10, 0.85), (1.10, 0.90), (1.10, 0.95),
        (1.15, 0.85), (1.15, 0.90), (1.15, 0.95)
    ]

    total_data_length = 0
    indicators = {}
    ground_truth = {}
    for symbol in intrinsic_events:
        path = f"cache/indicators"
        with open(f"{path}/{symbol}.pickle", 'rb') as f:
            indicators[symbol] = pickle.load(f)

        data_length = indicators[symbol]['indicators'].shape[1]
        total_data_length += data_length

        ground_truth[symbol] = np.empty((len(limits), data_length))

        for limit_idx, (take_profit, stop_loss) in enumerate(limits):
            steps = np.array(intrinsic_events[symbol]['prices'])
            profits = mark_steps(steps=steps, take_profit=take_profit, stop_loss=stop_loss)

            direction_idx = 0
            for indicator_idx in range(data_length):
                while intrinsic_events[symbol]['timestamps'][direction_idx] < indicator_idx:
                    direction_idx += 1
                ground_truth[symbol][limit_idx, indicator_idx] = profits[direction_idx]

            #fig, axs = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]})
            #axs[0].plot(steps)
            #axs[0].set_yscale('log')
            #axs[1].plot(profits)
            #plt.show()
            #break
        break

    skip_length = 60 * 24 * 30
    data_input = np.empty((7, total_data_length))
    data_output = np.empty((len(limits), total_data_length))
    data_offset = 0
    for symbol in indicators:
        data_length = indicators[symbol]['indicators'].shape[1] - skip_length
        if data_length <= 0:
            continue
        data_input[:, data_offset:data_offset + data_length] = indicators[symbol]['indicators'][:, skip_length:]
        data_output[:, data_offset:data_offset + data_length] = ground_truth[symbol][:, skip_length:]
        data_offset += data_length

    with open(f"cache/training_data.pickle", 'wb') as f:
        pickle.dump({
            'limits': limits,
            'skip_length': skip_length,
            'input': data_input,
            'output': data_output
        }, f)

    print("done")


if __name__ == '__main__':
    main()
