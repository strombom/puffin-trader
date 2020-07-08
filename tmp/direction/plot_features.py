
import matplotlib.pyplot as plt
import torch

import torch as __torch__

tensor = None

for params in torch.jit.load('../' + 'observations.dat_tensor').cpu().parameters():
    tensor = params
    break

row_names = [
    'bitmex price max',
    'bitmex price min',
    'bitmex volume buy',
    'bitmex volume sell',

    'binance offset max',
    'binance offset min',
    'binance volume buy',
    'binance volume sell',

    'coinbase offset max',
    'coinbase offset min',
    'coinbase volume buy',
    'coinbase volume sell',
]

"""
for row in range(tensor.size(1)):
    print("")
    print("--- " + row_names[row] + " ---")
    for col in range(tensor.size(2)):
        
        print("lookback(" + str(col) + ") ", end = '')
        print(tensor[:, row, col].detach().numpy())
"""
#8532712
#print(tensor[8532571:8546990, 0:2, :])

for idx in range(8532571, 8546990):
    for row in range(2):
        for col in range(16):
            n = tensor[idx, row, col]
            if n > 9000 or n < -9000:
                tensor[idx, row, col] = 0.0

#print(tensor[8532571:8546990, 0:2, :])


import matplotlib.pyplot as plt
fig, axs = plt.subplots(4, 4, sharey=False, tight_layout=False)

for row in range(4):
    for col in range(4):
        data = tensor[:, 0, row * 4 + col]
        axs[row][col].hist(data, bins=50)

plt.show()


print(data)
print(data.size())
data = tensor[::2, 2, 4]
print(data.size())

print(tensor.size())



#for a in tensor.named_parameters():
#    print(a)

#for a in tensor.parameters():
#    print(a.size())
#    quit()

quit()



for tensor_name in ("past_observations", "future_positives", "future_negatives"):
    titles = ("Price", "Buy vol", "Sell vol")
    tensor = torch.load('data/' + tensor_name + '_0.tensor').cpu()
    B, C, N, O = 0, 1, 2, 3 # 160
    nrows, ncols = tensor.size()[C], tensor.size()[N]
    nrows, ncols = min(nrows, 3), min(ncols, 6)
    fig, ax_rows = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False)
    for row_idx, ax_row in enumerate(ax_rows):
        try:
            iter(ax_row)
        except:
            ax_row = [ax_row]
        for col_idx, ax in enumerate(ax_row):
            ax.plot(tensor[0, row_idx, col_idx, :])
            if row_idx == 0:
                ax.set_ylim(-1, 1)
            else:
                ax.set_ylim(0, 1)
            if col_idx == 0:
                ax.set_ylabel(titles[row_idx])

    fig.set_size_inches(ncols*4, nrows*4)
    fig.suptitle(tensor_name, fontsize=10)
    plt.savefig(tensor_name + ".png")

quit()

for tensor_name in ("past_observations", "future_positives", "future_negatives"):
    titles = ("T0", "T+10", "T+20")
    tensor = torch.load('data/' + tensor_name + '_0.tensor').cpu()
    B, C, N, O = 0, 1, 2, 3 # 2x3x4x160
    nrows, ncols = tensor.size()[B], tensor.size()[N]
    nrows, ncols = min(nrows, 3), min(ncols, 4)
    fig, ax_rows = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    for row_idx, ax_row in enumerate(ax_rows):
        for col_idx, ax in enumerate(ax_row):
            ax.plot(tensor[row_idx, 0, col_idx, :])
            ax.set_ylim(-1, 1)
            if col_idx == 0:
                ax.set_ylabel(titles[row_idx])
    fig.set_size_inches(ncols*4, nrows*4)
    fig.suptitle(tensor_name, fontsize=10)
    plt.savefig(tensor_name + ".png")

quit()
