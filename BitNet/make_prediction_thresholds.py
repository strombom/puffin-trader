import glob
import pickle
import numpy as np
import pandas as pd
from numba import jit
from datetime import timedelta
from Indicators.supersmoother import SuperSmoother


#@jit
def make_positions(ground_truth_timestamp_start, ground_truth_timestamp_end, ground_truths, timestamps, threshold, predictions):
    delta_idx = 1

    positions = []
    # thresholds[symbol] = []
    # p_y = []
    # th_y = []
    gt_idx = 0
    gt_up, gt_down = 0, 0
    for ts_idx, timestamp in enumerate(timestamps):
        if predictions[ts_idx] is not None:
            #while pd.to_datetime(ground_truth.iloc[gt_idx]['timestamp']) < timestamp:
            while ground_truth_timestamp_start[gt_idx] < timestamp:
                gt_idx += 1

            prediction = predictions[ts_idx][delta_idx]
            # threshold = supersmoother.append(prediction) + 0.25
            # threshold = 0.25
            if prediction > threshold:
                positions.append({
                    'start': timestamp,
                    'end': ground_truth_timestamp_end[gt_idx].to_pydatetime(),
                    'gt': ground_truths[ts_idx][delta_idx]
                })
                # p_y.append(min(1.0, max(-1.0, prediction)))
                # th_y.append(threshold)
                # thresholds[symbol].append(threshold)
                if ground_truths[ts_idx][delta_idx] > 0:
                    gt_up += 1
                else:
                    gt_down += 1
        # else:
        #    thresholds[symbol].append(None)

    return positions, gt_up, gt_down


def main():
    try:
        with open(f'cache/predictions.pickle', 'rb') as f:
            prediction_data = pickle.load(f)
            symbols = prediction_data['symbols']
            timestamps = prediction_data['timestamps']
            predictions = prediction_data['predictions']
            #prediction_indices = prediction_data['prediction_indices']
            ground_truths = prediction_data['ground_truths']
    except FileNotFoundError:
        print("Load predictions fail ")
        quit()

    #ground_truth.sort_values(by='ground_truth_timestamp', axis='index', ascending=True, inplace=True)
    symbols = ["ADAUSDT", "BCHUSDT", "BNBUSDT", "BTTUSDT", "CHZUSDT", "EOSUSDT", "ETCUSDT", "LINKUSDT", "MATICUSDT", "THETAUSDT", "XLMUSDT", "XRPUSDT"]
    print(symbols)
    #symbols = symbols[0:1]

    #timestamp_start = timestamps[0]
    #timestamp_next = timestamp_start + timedelta(hours=1)

    #supersmoother = SuperSmoother(period=500, initial_value=0)

    #thresholds = {}
    for symbol_idx, symbol in enumerate(symbols):

        ground_truth_path = 'E:/BitBot/simulation_data'
        ground_truth = pd.read_csv(
            ground_truth_path + f'/2020-07-01_{symbol}.csv',
            parse_dates=["timestamp", "ground_truth_timestamp"],
            index_col='ind_idx'
        )

        ground_truth_timestamp_start = ground_truth[['timestamp']].to_numpy().squeeze()
        ground_truth_timestamp_end = ground_truth[['ground_truth_timestamp']].to_numpy().squeeze()  # .iloc[gt_idx]

        #for threshold in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for threshold in [0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.4]:
            print(symbol_idx, symbol, threshold)
            #positions = np.empty()

            #def make_positions(ground_truth_timestamp_start, ground_truth_timestamp_end, ground_truths, timestamps,
            #                   threshold, predictions):

            positions, gt_up, gt_down = make_positions(
                ground_truth_timestamp_start=ground_truth_timestamp_start,
                ground_truth_timestamp_end=ground_truth_timestamp_end,
                ground_truths=ground_truths[symbol],
                timestamps=timestamps[symbol],
                threshold=threshold,
                predictions=predictions[symbol]
            )

            with open(f"cache/thresholds/positions_{symbol}_{threshold:.3f}.pickle", 'wb') as f:
                pickle.dump({
                    'positions': positions,
                    'gt_up': gt_up,
                    'gt_down': gt_down
                }, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(gt_up, gt_down, gt_up / (gt_up + gt_down))

    quit()

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(p_y, c='tab:orange', linewidth=0.1)
    ax.plot(th_y, c='tab:blue')
    plt.show()
    quit()

    """
    with open(f"cache/predictions_with_th.pickle", 'wb') as f:
        pickle.dump({
            'symbols': symbols,
            'timestamps': timestamps,
            'predictions': predictions,
            'ground_truths': ground_truths,
            #'prediction_indices': prediction_indices,
            'thresholds': thresholds
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    delta_idx = 12
    bin_size = 200
    threshold_min, threshold_max = int(bin_size * 0.5625), int(bin_size * 0.85)
    threshold_limit = 5
    history_length = 1000
    symbol_count = len(symbols)

    buffer_idx = 0
    buffer = np.zeros(shape=(history_length, symbol_count), dtype=int) - 1
    bin_sum = np.zeros(shape=bin_size, dtype=int)
    top_idx = threshold_min

    thresholds = {}

    for symbol_idx, symbol in enumerate(symbols):
        thresholds[symbol] = []
        p_y = []
        th_y = []
        for idx in range(len(timestamps)):
            if predictions[symbol][idx] is not None:
                remove_idx = buffer[buffer_idx, symbol_idx]
                if remove_idx != -1:
                    bin_sum[remove_idx] -= 1
                    if remove_idx == top_idx:
                        while bin_sum[top_idx] == 0 and top_idx > threshold_min:
                            top_idx -= 1

                prediction = int((predictions[symbol][idx][delta_idx] + 2) * 50)
                threshold_idx = max(0, min(bin_size - 1, prediction))
                buffer[buffer_idx, symbol_idx] = threshold_idx
                buffer_idx = (buffer_idx + 1) % history_length

                bin_sum[threshold_idx] += 1
                top_idx = max(top_idx, threshold_idx)

                threshold_sum = 0
                threshold_idx = top_idx
                while threshold_sum < threshold_limit and threshold_idx > threshold_min:
                    threshold_sum += bin_sum[threshold_idx]
                    threshold_idx -= 1
                threshold_idx = min(threshold_idx, threshold_max)
                threshold = 2 * (threshold_idx * 2 - bin_size) / bin_size
                thresholds[symbol].append({'timestamp': timestamps[idx], 'threshold': threshold})

                p_y.append(min(1.0, max(-1.0, predictions[symbol][idx][delta_idx])))
                th_y.append(threshold)

        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(p_y, c='tab:orange', linewidth=0.1)
        #ax.plot(th_y, c='tab:blue')
        plt.show()
        quit()
        print(1)
    """

    """
    chunk_size = 100
    bin_size = 200
    delta_idx = 12
    thresholds = {}

    for symbol in symbols[0:2]:
        thresholds[symbol] = []
        chunks = []
        chunk = np.zeros(bin_size)
        counter = 0
        for idx in range(len(timestamps)):
            if predictions[symbol][idx] is not None:
                chunk[int((predictions[symbol][idx][delta_idx] + 2) * 50)] += 1
                counter += 1

            if counter == 100:
                chunks.append(chunk)
                chunk = np.zeros(bin_size)
                counter = 0

                if len(chunks) == 50:
                    sum_chunk = chunks[0]
                    for chunk in chunks[1:]:
                        sum_chunk += chunk
                    threshold_sum = 0
                    threshold = -1
                    for c_idx in range(bin_size - 1, -1, -1):
                        threshold_sum += sum_chunk[c_idx]
                        if threshold_sum >= 5:
                            threshold = c_idx
                            break
                    threshold = max(100, min(125, threshold))
                    threshold = 2 * (threshold * 2 - bin_size) / bin_size
                    thresholds[symbol].append({'timestamp': timestamps[idx], 'threshold': threshold})
                    chunks = chunks[1:]
    """
    print(1)


if __name__ == '__main__':
    main()
