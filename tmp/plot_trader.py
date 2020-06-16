

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import sys
import csv

start = int(sys.argv[1])
try:
    stop = int(sys.argv[2])
except:
    stop = start


for episode_idx in range(start, stop + 1):

    for episode_type in ("val", "train"):

        titles = []
        rows = []

        with open('C:\\development\\github\\puffin-trader\\tmp\\log\\bitmex_' + episode_type + '_' + str(episode_idx) + '.csv') as csvfile:
            reader = csv.reader(csvfile)
            is_first = True
            for row in reader:
                if is_first:
                    titles = row
                    is_first = False
                else:
                    rows.append([float(i) for i in row])


        rows = np.array(rows)
        x_range = np.arange(rows.shape[0])
        values = (rows[:,1] + rows[:,2])

        def make_patch_spines_invisible(ax):
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)

        def center_y(ax):
            yabs_max = abs(max(ax.get_ylim(), key=abs))
            ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)


        fig, (host_price, host_leverage, host_upnl, host_value) = plt.subplots(4, 1, figsize=(14, 8), sharex='col', gridspec_kw={'hspace': 0})
        fig.subplots_adjust(top=0.95, bottom=0.05, left=0.06, right=0.97)


        price_points = np.array([x_range, rows[:,0]]).T.reshape(-1, 1, 2)
        price_segments = np.concatenate([price_points[:-1], price_points[1:]], axis=1)
        cmap = ListedColormap(['r', 'b', 'g'])
        norm = BoundaryNorm([-10, -1, 1, 10], cmap.N)
        lc = LineCollection(price_segments, cmap=cmap, norm=norm)
        lc.set_array(rows[:,4])
        lc.set_linewidth(2)
        line = host_price.add_collection(lc)
        host_price.set_ylim(rows[:,0].min() - 10, rows[:,0].max() + 10)


        plot_pos_leverage = host_leverage.fill_between(x_range, 0, rows[:,4], color='tab:blue', alpha=0.8, label="")
        plot_order_leverage = host_leverage.fill_between(x_range, 0, rows[:,6], color='tab:orange', alpha=0.15, label="")

        plot_upnl, = host_upnl.plot(rows[:,2], color='tab:green', label="")
        plot_wallet, = host_value.plot(rows[:,1] + rows[:,2], color='tab:blue', label="")
        #plot_value, = host_value.plot(values, color='tab:red', label="")

        host_price.yaxis.grid(linestyle='dashed')
        center_y(host_upnl)
        host_upnl.yaxis.grid(linestyle='dashed')
        host_value.yaxis.grid(linestyle='dashed')

        host_value.set_xlabel("Time")

        host_price.set_ylabel("Price")

        host_leverage.set_ylabel("Leverage")
        host_upnl.set_ylabel("Unrealised PnL")
        host_value.set_ylabel("Wallet + UPnL")

        host_price.yaxis.label.set_color('tab:blue')
        host_leverage.yaxis.label.set_color('tab:blue')
        host_upnl.yaxis.label.set_color(plot_upnl.get_color())
        host_value.yaxis.label.set_color(plot_wallet.get_color())

        """
        host_price_line = host_price.axvline(x=0, color="k", linewidth=0.2)
        host_leverage_line = host_leverage.axvline(x=0, color="k", linewidth=0.2)
        host_upnl_line = host_upnl.axvline(x=0, color="k", linewidth=0.2)
        host_value_line = host_value.axvline(x=0, color="k", linewidth=0.2)

        def onMouseMove(event):
          host_price_line.set_data([event.xdata, event.xdata], [0, 1])
          host_leverage_line.set_data([event.xdata, event.xdata], [0, 1])
          host_upnl_line.set_data([event.xdata, event.xdata], [0, 1])
          host_value_line.set_data([event.xdata, event.xdata], [0, 1])
          fig.canvas.draw()

        fig.canvas.mpl_connect('motion_notify_event', onMouseMove)
        """

        fig.savefig('trading\\trade_' + episode_type + '_' + str(episode_idx) + '.png', dpi=140)
        plt.close()
        #plt.show()
