
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv(
        "Data/BTCUSDT2021-11-27.csv",
    )[::-1].reset_index()

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    fig, ax = plt.subplots()
    ax.plot(df['price'])
    #ax.plot(df['timestamp'], df['price'])
    plt.show()


