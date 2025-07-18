import os
import glob
import torch
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm


def calc_up_down(p):
    ups_, downs_ = (p > 0.0).nonzero().squeeze(), (p < 0.0).nonzero().squeeze()
    up_count, down_count = 0, 0
    if len(downs_.shape) > 0:
        down_count = downs_.shape[0]
    if len(ups_.shape) > 0:
        up_count = ups_.shape[0]
    return up_count, down_count


def print_up_down(up_count, down_count):
    n = 4
    if down_count + up_count == 0:
        print(f"  {up_count:n} up, {down_count:n} down, tot {(up_count + down_count):n}, ratio -", end='')
    else:
        print(
            f"  {up_count:n} up, {down_count:n} down, tot {(up_count + down_count):n}, ratio {up_count / (up_count + down_count):.2f}",
            end='')


def main():
    bins = 20
    stats, dates, all_stats = [], [], {}
    ptm, pvm, gttm, gtvm, cnt = 0, 0, 0, 0, 0
    y_idx = 12

    training_path = 'E:/BitBot/simulation_data/'

    bin_count = 200

    if True:
        file_path = f"cache/predictions.pickle"
        try:
            with open(file_path, 'rb') as f:
                prediction_data = pickle.load(f)
                symbols = prediction_data['symbols']
                timestamps = prediction_data['timestamps']
                predictions = prediction_data['predictions']
                prediction_indices = prediction_data['prediction_indices']
                ground_truths = prediction_data['ground_truths']
                thresholds = prediction_data['thresholds']
        except FileNotFoundError:
            print("Load predictions fail ", file_path)
            quit()

        #symbols.remove('FTMUSDT')

        timestamp_start = timestamps[0].replace(hour=0, minute=0, second=0)
        timestamp_end = timestamps[-1].replace(hour=0, minute=0, second=0)
        daily_bins = []
        for day_idx in range((timestamp_end - timestamp_start).days):
            daily_bins.append({
                'up': [0 for n in range(bin_count)],
                'down': [0 for n in range(bin_count)]
            })

        for symbol in symbols:
            for idx in prediction_indices[symbol]:
                day_idx = (timestamps[idx] - timestamp_start).days
                bin_idx = int((predictions[symbol][idx][y_idx] + 2) * 50)
                if 0 <= bin_idx < bin_count:
                    if ground_truths[symbol][idx][y_idx] == 1:
                        daily_bins[day_idx]['up'][bin_idx] += 1
                    else:
                        daily_bins[day_idx]['down'][bin_idx] += 1

        file_path = f"cache/bins.pickle"
        with open(file_path, 'wb') as f:
            pickle.dump(daily_bins, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        file_path = f"cache/bins.pickle"
        with open(file_path, 'rb') as f:
            daily_bins = pickle.load(f)

    thresholds_buffer = []
    thresholds = []
    for daily_bin in daily_bins:
        threshold = (sum(daily_bin['up']) + sum(daily_bin['down'])) * 0.0005
        threshold_idx = 125
        binsum = 0
        for idx in range(bin_count - 1, -1, -1):
            binsum += daily_bin['up'][idx] + daily_bin['down'][idx]
            if binsum >= threshold:
                threshold_idx = min(idx, threshold_idx)
                break
                #if len(thresholds) > 2:
                #    threshold_idx = min(idx, )
                #else:
                #    threshold_idx = min(idx, threshold_idx)
                #break
        thresholds_buffer.append(threshold_idx)
        thresholds.append(thresholds_buffer[max(0, len(thresholds_buffer) - 3)])

    up_bins, down_bins = np.zeros(bin_count), np.zeros(bin_count)
    for day_idx in range(0, len(daily_bins)):
        for bin_idx in range(thresholds[day_idx], int(bin_count * 0.85)):
            up_bins[bin_idx] += daily_bins[day_idx]['up'][bin_idx]
            down_bins[bin_idx] += daily_bins[day_idx]['down'][bin_idx]
    tot_bins = up_bins + down_bins

    for bin_idx in range(bin_count - 1, -1, -1):
        print(f"Bin {bin_idx / 50 - 2:2.3f}: ", end='')
        print_up_down(up_count=up_bins[bin_idx], down_count=down_bins[bin_idx])
        print()

    """
    import matplotlib.pyplot as plt
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    plt.plot(up_bins)
    plt.plot(down_bins)
    plt.plot(thresholds)
    plt.show()
    """

    print(sum(tot_bins[130:]))
    print(sum(tot_bins))

    x = list(range(len(daily_bins)))
    y = 2 - np.array(list(range(bin_count))) / bin_count * 4
    z_val = np.empty((len(y), len(x)))

    for day_idx, day_bins in enumerate(daily_bins):
        for bin_idx in range(bin_count):
            weight = day_bins['up'][bin_idx] + day_bins['down'][bin_idx]
            if weight == 0:
                val = 0
            else:
                val = (day_bins['up'][bin_idx] - day_bins['down'][bin_idx]) / (day_bins['up'][bin_idx] + day_bins['down'][bin_idx])
            z_val[bin_idx, day_idx] = val

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    fig = plt.figure()
    ax = fig.add_subplot(111)  # , projection='3d')
    ax.pcolormesh(z_val, cmap=cm.coolwarm)
    #ax.plot_surface(X, Y, z_weight, cmap=cm.coolwarm)
    ax.plot(np.arange(len(daily_bins)) + 0.5, thresholds)
    plt.show()


    #print(up_bins)
    #print(down_bins)

    quit()

    # Load ground truth
    for file_path in glob.glob(training_path + "*.csv"):
        symbol = file_path.split("_")[-1].replace(".csv", "")
        data = pd.read_csv(file_path, index_col='ind_idx', parse_dates=['timestamp'], infer_datetime_format=True)
        print(data)

        break

    quit()
    started = False
    for file_path in glob.glob("preds/preds_*.pickle"):
        #if "2020-01-07" in file_path:
        #    break

        if "2020-01-15" in file_path:
            started = True
        #if "2020-05-15" in file_path:
        #    break
        if not started:
            continue

        date = file_path.split('_')[1].split('.')[0]
        dates.append(date)

        print(f"{file_path}    ", end='')

        with open(file_path, 'rb') as f:
            predictions = pickle.load(f)

        pred_train_mean = predictions['pred_train'][:, y_idx].mean()
        gt_train_mean = predictions['gt_train'][:, y_idx].double().mean()
        pred_val_mean = predictions['pred_val'][:, y_idx].mean()
        gt_val_mean = predictions['gt_val'][:, y_idx].double().mean()
        print(f"pred_train_mean {pred_train_mean:.3f},  gt_train_mean {gt_train_mean:.3f},  pred_val_mean {pred_val_mean:.3f},  gt_val_mean {gt_val_mean:.3f}")

        ptm += pred_train_mean
        pvm += pred_val_mean
        gttm += gt_train_mean
        gtvm += gt_val_mean
        cnt += 1

        stat = {}
        for i in range(bins - 1, -bins - 1, -1):
            minval_a = (i + 0) / bins * 2
            minval_b = (i + 1) / bins * 2

            #minval_a = gt_train_mean - (i + 1) / 20
            #minval_b = gt_train_mean - (i + 0) / 20

            gt_train_mean = 0

            bigs_train = torch.logical_and(
                predictions['pred_train'][:, y_idx] - gt_train_mean > minval_a,
                predictions['pred_train'][:, y_idx] - gt_train_mean < minval_b
            ).nonzero().squeeze()

            bigs_val = torch.logical_and(
                predictions['pred_val'][:, y_idx] - gt_train_mean > minval_a,
                predictions['pred_val'][:, y_idx] - gt_train_mean < minval_b
            ).nonzero().squeeze()

            p_train = predictions['gt_train'][:, y_idx][bigs_train]
            p_val = predictions['gt_val'][:, y_idx][bigs_val]

            #print(f"minval {minval_a: 2.2f} - {minval_b: 2.2f}      ", end='')

            if minval_a not in stat:
                stat[minval_a] = {
                    'train_up': 0,
                    'train_down': 0,
                    'val_up': 0,
                    'val_down': 0
                }

            if minval_a not in all_stats:
                all_stats[minval_a] = {
                    'train_up': 0,
                    'train_down': 0,
                    'val_up': 0,
                    'val_down': 0
                }

            ups, downs = calc_up_down(p_train)
            stat[minval_a]['train_up'] += ups
            stat[minval_a]['train_down'] += downs
            #print_up_down(ups, downs)
            #print("     ", end='')
            all_stats[minval_a]['train_up'] += ups
            all_stats[minval_a]['train_down'] += downs

            ups, downs = calc_up_down(p_val)
            stat[minval_a]['val_up'] += ups
            stat[minval_a]['val_down'] += downs
            #print_up_down(ups, downs)
            #print()
            all_stats[minval_a]['val_up'] += ups
            all_stats[minval_a]['val_down'] += downs

        stats.append(stat)
        #break

    x = list(range(len(dates)))
    y = 1 - np.array(list(range(len(stats[0])))) / len(stats[0]) * 2
    z_val = np.empty((len(y), len(x)))
    z_weight = np.empty((len(y), len(x)))

    for stat_idx, stat in enumerate(stats):
        for pred_idx, pred in enumerate(stat):
            weight = stat[pred]['val_up'] + stat[pred]['val_down']
            if weight == 0:
                val = 0
            else:
                val = stat[pred]['val_up'] / (stat[pred]['val_up'] + stat[pred]['val_down']) - 0.5
            z_val[pred_idx, stat_idx] = val
            z_weight[pred_idx, stat_idx] = weight

    X, Y = np.meshgrid(x, y)
    #Z = np.reshape(z_val, X.shape)  # Z.shape must be equal to X.shape = Y.shape

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    fig = plt.figure()
    ax = fig.add_subplot(111)  # , projection='3d')
    ax.pcolormesh(z_val, cmap=cm.coolwarm)
    #ax.plot_surface(X, Y, z_weight, cmap=cm.coolwarm)
    plt.show()

    ptm /= cnt
    pvm /= cnt
    gttm /= cnt
    gtvm /= cnt
    print(f"pred_train_mean {ptm:.3f},  gt_train_mean {gttm:.3f},  pred_val_mean {pvm:.3f},  gt_val_mean {gtvm:.3f}")

    for minval_a in all_stats:
        if minval_a == -0.05:
            print()

        print(f"minval {minval_a + 0.05: 2.2f} - {minval_a: 2.2f}      ", end='')
        print_up_down(all_stats[minval_a]['train_up'], all_stats[minval_a]['train_down'])
        print("     ", end='')
        print_up_down(all_stats[minval_a]['val_up'], all_stats[minval_a]['val_down'])
        print()


if __name__ == '__main__':
    main()
