import os
import glob


if __name__ == '__main__':

    for file_path in glob.glob("cache/tickers/*.csv"):
        pair = os.path.basename(file_path).replace('.csv', '')
        print(pair, file_path)


