
import math


def transform(x):
    x = abs(x) ** (1. / 10)
    if x > 0.95:
        x -= 0.95
    x *= 0.2
    return x



"""
for i in range(100):
    a = i / 100.0
    b = transform(a)
    print(a, b)

quit()
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle


channel = 2
rows = []
for tensor_name in ("past_observations", "future_positives", "future_negatives"):
    for idx in range(10):
        tensor = torch.load("data/" + tensor_name + "_" + str(idx) + ".tensor").cpu()
        for batch in range(tensor.size()[0]):
            for oidx in range(tensor.size()[2]):
                obs = tensor[batch][channel][oidx].numpy()
                rows.append(obs)
                #omin = transform(obs.min())
                #omax = transform(obs.max())
                #rows.append([omin, omax])
pickle.dump(rows, open("observations.pickle", "wb"))


rows = pickle.load(open("observations.pickle", "rb"))

rows = np.array(rows)
rows = rows.flatten()
rows = rows[0:1000000]

for i in range(rows.shape[0]):
    rows[i] = transform(rows[i])

plt.hist(rows, bins=1000)
plt.show()

quit()

"""
ex_min, ex_max = 0, 0
rows_min, rows_max = [], []

rows = []
for tensor_name in ("past_observations", "future_positives", "future_negatives"):
    for idx in range(10):
        tensor = torch.load("data/" + tensor_name + "_" + str(idx) + ".tensor").cpu()
        for batch in range(tensor.size()[0]):
            for oidx in range(tensor.size()[2]):
                obs = tensor[batch][0][oidx].numpy()

                if obs.min() < ex_min:
                    ex_min = obs.min()
                    rows_min.append(obs)
                if obs.max() > ex_max:
                    ex_max = obs.max()
                    rows_max.append(obs)

                #rows.append([obs.min(), obs.max()])
#rows = np.array(rows)
rows_min = np.array(rows_min)
rows_max = np.array(rows_max)
"""



for idx, row in enumerate(rows_min):

    fig, ax = plt.subplots()

    ax.set_ylim(-0.25, 0.25)
    ax.plot(row)
    plt.savefig("min_" + str(idx) + ".png")
    plt.close()
    #plt.show()

for idx, row in enumerate(rows_max):

    fig, ax = plt.subplots()

    ax.set_ylim(-0.25, 0.25)
    ax.plot(row)
    plt.savefig("max_" + str(idx) + ".png")
    plt.close()
    #plt.show()



quit()

pickle.dump((rows_min, rows_max), open("observations.pickle", "wb"))
rows_min, rows_max = pickle.load(open("observations.pickle", "rb"))

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


print(rows_max.shape)
quit()

rows = rows.flatten()

print(rows.max(), rows.min())

plt.hist(rows, bins=250)
plt.show()

quit()


"""
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
"""
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


