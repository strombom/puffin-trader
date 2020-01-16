
import matplotlib.pyplot as plt
import torch

tensor = torch.load("x.zip")

# 2x3x4x256
B, C, N, F = 0, 1, 2, 3

nrows, ncols = tensor.size()[B], tensor.size()[N]

#nrows = min(nrows, 2)
#ncols = min(ncols, 4)

fig, ax_rows = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(6, 6))

for row_idx, ax_row in enumerate(ax_rows):
    for col_idx, ax in enumerate(ax_row):
        ax.plot(tensor[row_idx, 0, col_idx, :])

plt.show()

