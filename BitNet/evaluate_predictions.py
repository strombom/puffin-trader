
import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt


def main():
    def print_up_down(p):
        ups = (p > 0.0).nonzero().squeeze()
        downs = (p < 0.0).nonzero().squeeze()

        if len(downs.shape) == 0:
            down_count = 0
        else:
            down_count = downs.shape[0]

        if len(ups.shape) == 0:
            up_count = 0
        else:
            up_count = ups.shape[0]

        if down_count + up_count == 0:
            print(f"  {up_count} up, {down_count} down, tot {up_count + down_count}, ratio -", end='')
        else:
            print(f"  {up_count} up, {down_count} down, tot {up_count + down_count}, ratio {up_count / (up_count + down_count):.2f}", end='')

    with open('preds_2020-03-22.pickle', 'rb') as f:
        predictions = pickle.load(f)

    pred_train_mean = predictions['pred_train'].mean()
    gt_train_mean = predictions['gt_train'].double().mean()
    pred_val_mean = predictions['pred_val'].mean()
    gt_val_mean = predictions['gt_val'].double().mean()
    print(f"pred_train_mean {pred_train_mean:.3f},  gt_train_mean {gt_train_mean:.3f},  pred_val_mean {pred_val_mean:.3f},  gt_val_mean {gt_val_mean:.3f}")

    for i in range(-10, 10):
        minval_a = (i + 0) / 10
        minval_b = (i + 1) / 10

        #minval_a = gt_train_mean - (i + 1) / 20
        #minval_b = gt_train_mean - (i + 0) / 20

        print(f"minval {minval_a: 2.2f} - {minval_b: 2.2f}      ", end='')

        gt_train_mean = 0

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

        print_up_down(p_train)
        print("     ", end='')
        print_up_down(p_val)
        print()


if __name__ == '__main__':
    main()
