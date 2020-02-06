
import csv
import time
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_rate(filename):
    with open(filename) as file:
        reader = csv.reader(file, delimiter=',')
        titles = next(reader)
        rows = []
        for row in reader:
            rows.append([float(item) for item in row])

        if len(rows) == 0:
            return

        data = np.array(rows)
        x = np.arange(0, len(rows))

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        if len(titles) == 1:
            fig, ax1 = plt.subplots()
            axes = [ax1]
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            axes = [ax1, ax2]
            for i in range(2, len(titles)):
                ax = ax2.twinx()
                axes.append(ax)

        for col_idx, ax in enumerate(axes):
            ax.set_ylabel(titles[col_idx])
            ax.yaxis.label.set_color(colors[col_idx])
            ax.tick_params(axis='y', colors=colors[col_idx], size=4, width=1.5)
            #if col_idx > 1:
            #    ax.spines["right"].set_position(("axes", 1.15 * (col_idx - 1)))
            width = 1
            ax.plot(x, data[:, col_idx], color=colors[col_idx], linewidth=width)
        
        fig.tight_layout()
        fig.savefig('learning_rate.png', dpi=140)


def read_file(filename):
    with open(filename, 'r') as file:
        return file.readlines()

filename = 'learning_rate.csv'
initial = ""
while True:
    current = read_file(filename=filename)
    if initial != current:
        plot_learning_rate(filename=filename)
        initial = current

    time.sleep(0.5)
