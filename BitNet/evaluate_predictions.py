import os
import glob

import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


def main():
    def calc_up_down(p):
        ups_ = (p > 0.0).nonzero().squeeze()
        downs_ = (p < 0.0).nonzero().squeeze()

        if len(downs_.shape) == 0:
            down_count = 0
        else:
            down_count = downs_.shape[0]

        if len(ups_.shape) == 0:
            up_count = 0
        else:
            up_count = ups_.shape[0]

        return up_count, down_count

    def print_up_down(up_count, down_count):
        if down_count + up_count == 0:
            print(f"  {up_count} up, {down_count} down, tot {up_count + down_count}, ratio -", end='')
        else:
            print(f"  {up_count} up, {down_count} down, tot {up_count + down_count}, ratio {up_count / (up_count + down_count):.2f}", end='')

    bins = 20

    stats = []
    dates = []
    all_stats = {}
    y_idx = 1

    ptm = 0
    pvm = 0
    gttm = 0
    gtvm = 0
    cnt = 0

    started = False
    for file_path in glob.glob("preds/preds_*.pickle"):
        #if "2020-01-07" in file_path:
        #    break

        #if "2020-03-01" in file_path:
        #    started = True
        #if "2020-03-27" in file_path:
        #    break
        #if not started:
        #    continue

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
            minval_a = (i + 0) / bins
            minval_b = (i + 1) / bins

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
