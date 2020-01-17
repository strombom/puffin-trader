
import matplotlib.pyplot as plt
import torch

for tensor_name in ("past_observations", "future_positives", "future_negatives"):
    titles = ("T0", "T+10", "T+20")
    tensor = torch.load(tensor_name + ".tensor")
    B, C, N, O = 0, 1, 2, 3 # 2x3x4x160
    nrows, ncols = tensor.size()[B], tensor.size()[N]
    fig, ax_rows = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    for row_idx, ax_row in enumerate(ax_rows):
        for col_idx, ax in enumerate(ax_row):
            ax.plot(tensor[row_idx, 0, col_idx, :])
            ax.set_ylim(-0.02, 0.02)
            if col_idx == 0:
                ax.set_ylabel(titles[row_idx])
    fig.set_size_inches(ncols*4, nrows*4)
    fig.suptitle(tensor_name, fontsize=10)
    plt.savefig(tensor_name + ".png")

quit()

for tensor_name in ("past_observations", "future_positives", "future_negatives"):
    titles = ("Price", "Buy vol", "Sell vol")
    tensor = torch.load(tensor_name + ".tensor")
    B, C, N, O = 0, 1, 2, 3 # 160
    nrows, ncols = tensor.size()[C], tensor.size()[N]
    fig, ax_rows = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False)
    for row_idx, ax_row in enumerate(ax_rows):
        for col_idx, ax in enumerate(ax_row):
            ax.plot(tensor[0, row_idx, col_idx, :])
            if row_idx == 0:
                ax.set_ylim(-0.02, 0.02)
            else:
                ax.set_ylim(0, 15000000)
            if col_idx == 0:
                ax.set_ylabel(titles[row_idx])

    fig.set_size_inches(ncols*4, nrows*4)
    fig.suptitle(tensor_name, fontsize=10)
    plt.savefig(tensor_name + ".png")


