
import matplotlib.pyplot as plt
import numpy as np
import csv


titles = []
rows = []

with open('C:\\development\\github\\puffin-trader\\tmp\\simulation.csv') as csvfile:
    reader = csv.reader(csvfile)
    is_first = True
    for row in reader:
        if is_first:
            titles = row
            is_first = False
        else:
            rows.append([float(i) for i in row])


rows = np.array(rows)

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

fig, host = plt.subplots(1, 1, figsize=(12, 10))
fig.subplots_adjust(right=0.65)

par1 = host.twinx()
par2 = host.twinx()
par3 = host.twinx()

par2.spines["right"].set_position(("axes", 1.2))
make_patch_spines_invisible(par2)
par2.spines["right"].set_visible(True)

par3.spines["right"].set_position(("axes", 1.4))
make_patch_spines_invisible(par3)
par3.spines["right"].set_visible(True)

p1, = host.plot(rows[:,0], "b-", label="Last price")
p2, = par1.plot(rows[:,2], "g-", label="Contracts")
p3, = par2.plot(rows[:,3], "r-", label="Wallet")
p4, = par3.plot(rows[:,4], "y-", label="UPNL")

host.set_xlabel("Time")
host.set_ylabel("Last price")
par1.set_ylabel("Contracts")
par2.set_ylabel("Wallet")
par3.set_ylabel("UPNL")

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())
par3.yaxis.label.set_color(p4.get_color())


#host.set_xlim(0, 2)
#host.set_ylim(0, 2)
#par1.set_ylim(0, 4)
#par2.set_ylim(1, 65)
#par3.set_ylim(1, 65)
#par4.set_ylim(1, 65)



plt.show()

quit()



for col_idx in len(titles):
    ax.plot(rows[col_idx])


print(titles)
print(rows)

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
