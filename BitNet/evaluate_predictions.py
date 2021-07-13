import glob

import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt


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

    stats = {}

    ptm = 0
    pvm = 0
    gttm = 0
    gtvm = 0
    cnt = 0

    for file_path in glob.glob("preds_*.pickle"):
        print(f"{file_path}    ", end='')

        with open(file_path, 'rb') as f:
            predictions = pickle.load(f)

        pred_train_mean = predictions['pred_train'].mean()
        gt_train_mean = predictions['gt_train'].double().mean()
        pred_val_mean = predictions['pred_val'].mean()
        gt_val_mean = predictions['gt_val'].double().mean()
        print(f"pred_train_mean {pred_train_mean:.3f},  gt_train_mean {gt_train_mean:.3f},  pred_val_mean {pred_val_mean:.3f},  gt_val_mean {gt_val_mean:.3f}")

        ptm += pred_train_mean
        pvm += pred_val_mean
        gttm += gt_train_mean
        gtvm += gt_val_mean
        cnt += 1

        for i in range(-10, 10):
            minval_a = (i + 0) / 10
            minval_b = (i + 1) / 10

            if minval_a not in stats:
                stats[minval_a] = {
                    'train_up': 0,
                    'train_down': 0,
                    'val_up': 0,
                    'val_down': 0
                }

            #minval_a = gt_train_mean - (i + 1) / 20
            #minval_b = gt_train_mean - (i + 0) / 20

            gt_train_mean = 0.08

            bigs_train = torch.logical_and(
                predictions['pred_train'] - gt_train_mean > minval_a,
                predictions['pred_train'] - gt_train_mean < minval_b
            ).nonzero().squeeze()

            bigs_val = torch.logical_and(
                predictions['pred_val'] - gt_train_mean > minval_a,
                predictions['pred_val'] - gt_train_mean < minval_b
            ).nonzero().squeeze()

            p_train = predictions['gt_train'][bigs_train]
            p_val = predictions['gt_val'][bigs_val]

            #print(f"minval {minval_a: 2.2f} - {minval_b: 2.2f}      ", end='')

            ups, downs = calc_up_down(p_train)
            stats[minval_a]['train_up'] += ups
            stats[minval_a]['train_down'] += downs
            #print_up_down(ups, downs)
            #print("     ", end='')

            ups, downs = calc_up_down(p_val)
            stats[minval_a]['val_up'] += ups
            stats[minval_a]['val_down'] += downs
            #print_up_down(ups, downs)
            #print()

    ptm /= cnt
    pvm /= cnt
    gttm /= cnt
    gtvm /= cnt
    print(f"pred_train_mean {ptm:.3f},  gt_train_mean {gttm:.3f},  pred_val_mean {pvm:.3f},  gt_val_mean {gtvm:.3f}")

    for minval_a in stats:
        if minval_a == 0:
            print()

        print(f"minval {minval_a: 2.2f} - {minval_a + 0.1: 2.2f}      ", end='')
        print_up_down(stats[minval_a]['train_up'], stats[minval_a]['train_down'])
        print("     ", end='')
        print_up_down(stats[minval_a]['val_up'], stats[minval_a]['val_down'])
        print()


if __name__ == '__main__':
    main()
