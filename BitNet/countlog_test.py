
import pandas as pd

counts = pd.read_csv('log/countlog_2021-10-08 234454.txt')

occ = 0
prev_count = 0
for idx, row in counts.iterrows():
    count = row['count']
    if count == 4 and prev_count == 3:
        print(row['timestamp'])
        occ += 1

    prev_count = count

print(occ)
