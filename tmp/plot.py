
import matplotlib.pyplot as plt
import torch

for tensor_name in ("past_observations", "future_positives", "future_negatives"):

    tensor = torch.load(tensor_name + ".tensor")
    print(tensor_name, tensor.size())

    # 2x3x4x256
    B, C, N, F = 0, 1, 2, 3

    titles = ("Price", "Buy vol", "Sell vol")

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

quit()


for tensor_name in ("past_observations", "future_positives", "future_negatives"):

    tensor = torch.load(tensor_name + ".tensor")
    print(tensor_name, tensor.size())

    # 2x3x4x256
    B, C, N, F = 0, 1, 2, 3

    nrows, ncols = tensor.size()[B], tensor.size()[N]

    fig, ax_rows = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)

    for row_idx, ax_row in enumerate(ax_rows):
        for col_idx, ax in enumerate(ax_row):
            ax.plot(tensor[row_idx, 0, col_idx, :])

    fig.set_size_inches(ncols*4, nrows*4)
    fig.suptitle(tensor_name, fontsize=10)
    plt.savefig(tensor_name + ".png")
