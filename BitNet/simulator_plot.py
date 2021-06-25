import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv(
        "log/simlog 2021-06-25 165607.txt",
        parse_dates=['date'],
        date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S%z")
    )

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(df['date'], df['equity'])
    ax1.set_yscale('log')
    ax2.plot(df['date'], df['cash'] / df['equity'])
    plt.show()


if __name__ == '__main__':
    main()
