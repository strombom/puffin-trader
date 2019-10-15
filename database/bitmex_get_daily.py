
# Download compressed daily archives from bitmex


import io
import csv
import glob
import gzip
import tables
import urllib.request
from datetime import datetime
from datetime import timedelta
from queue import Queue
from threading import Thread


class TickTable(tables.IsDescription):
    timestamp   = tables.UInt64Col (pos = 0)
    price       = tables.Float32Col(pos = 1)
    volume      = tables.Float32Col(pos = 2)
    buy         = tables.BoolCol   (pos = 3)

class SymbolTable(tables.IsDescription):
    name        = tables.StringCol(16, pos = 0)
    ts_start    = tables.UInt64Col(pos = 1)
    ts_stop     = tables.UInt64Col(pos = 2)


h5file = tables.open_file("bitmex.h5", mode="a", title="Bitmex")

try:
    ticks_group = h5file.create_group("/", 'ticks', 'Tick data')
except:
    ticks_group = h5file._get_node('/ticks')

try:
    symbols_table = h5file.create_table('/', 'symbols', SymbolTable)
except:
    symbols_table = h5file._get_node('/symbols')

tick_tables = {}
symbol_row_idxs = {}

def update_symbols():
    for row in symbols_table.iterrows():
        symbol_name  = row['name'].decode('utf-8')
        tick_tables[symbol_name] = h5file._get_node('/ticks/' + symbol_name)
        symbol_row_idxs[symbol_name] = row.nrow

update_symbols()

def add_symbol(symbol_name, timestamp):
    symbols_table.row['name'] = symbol_name
    symbols_table.row['ts_start'] = timestamp
    symbols_table.row['ts_stop'] = 0
    symbols_table.row.append()
    symbols_table.flush()
    tick_tables[symbol_name] = h5file.create_table('/ticks', symbol_name, TickTable)
    update_symbols()

def append_trade_data(trade_data):
    updated_tables = {}

    for row in trade_data[1:]:
        if not row:
            break
        symbol_name = row[1]
        timestamp   = datetime.strptime(row[0][:-3], '%Y-%m-%dD%H:%M:%S.%f')
        timestamp   = int(datetime.timestamp(timestamp) * 1000000) # Timestamp in microseconds
        price       = row[4]
        volume      = row[8]
        buy         = True if row[2] == "Buy" else False

        if symbol_name not in symbol_row_idxs:
            add_symbol(symbol_name, timestamp)

        symbol_row_idx = symbol_row_idxs[symbol_name]
        symbol = symbols_table[symbol_row_idx]

        if timestamp > symbol['ts_stop']:
            try:
                tick_table = tick_tables[symbol_name]
                tick_table.row['timestamp'] = timestamp
                tick_table.row['price'] = price
                tick_table.row['volume'] = volume
                tick_table.row['buy'] = buy
                tick_table.row.append()
                symbols_table.cols.ts_stop[symbol_row_idx] = timestamp
                updated_tables[symbol_name] = tick_table
            except:
                pass


    symbols_table.flush()
    for tablename in updated_tables:
        tick_tables[tablename].flush()


"""
import pickle
with open('trade_data.pickle', 'rb') as f:
    trade_data = pickle.load(f)

import pickle
with open('trade_data.pickle', 'wb') as f:
    pickle.dump(rows, f, pickle.HIGHEST_PROTOCOL)
"""

"""
csv_files = []
for file in glob.glob("bitmex_daily/*.csv"):
    csv_files.append(file)
print(csv_files)
"""

# https://public.bitmex.com/?prefix=data/trade/

try:
    date_string = str(symbols_table.attrs.LAST_BITMEX_FILE_DATE)
except:
    date_string = "20141121"



url = "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/xxxxxxxx.csv.gz"

def download_file(date):
    date_string = datetime.strftime(date, '%Y%m%d')
    date_url = url.replace("xxxxxxxx", date_string)
    url_request = urllib.request.Request(date_url)
    url_connect = urllib.request.urlopen(url_request)
    data_gzip = url_connect.read(10**10)
    data_raw = gzip.GzipFile(fileobj=io.BytesIO(data_gzip)).read().decode("utf-8") 
    reader = csv.reader(data_raw.split('\n'), delimiter=',')
    rows = []
    for row in reader:
        #print('\t'.join(row))
        rows.append(row)
    print("Downloaded", date_url)
    return rows


def downloader(q):
    date = datetime.strptime(date_string, '%Y%m%d')
    date_last  = datetime.strptime("20191009", '%Y%m%d')
    while date < date_last:
        date = date + timedelta(days = 1)
        trade_data = download_file(date)
        q.put((date, trade_data))
    q.put(None)

q = Queue(maxsize=4)
worker = Thread(target = downloader, args = (q,))
worker.setDaemon(True)
worker.start()

while True:
    trade_data = q.get()
    if not trade_data:
        break
    date, trade_data = trade_data
    append_trade_data(trade_data)
    date_string = int(date.strftime("%Y%m%d"))
    print("Appended", date)
    symbols_table.attrs.LAST_BITMEX_FILE_DATE = date_string


h5file.close()
quit()


"""
2015-11-24D17:19:40.331762000   XBTZ15  Buy     1       342     ZeroPlusTick    d3253a44-71bb-010d-b383-e60b6b4dac48   342000                                                                                                                   0.00342  1.16964
2015-11-24D01:13:28.846232000   XBUZ15  Sell    5       324.57  PlusTick        1e619051-6aba-06ef-cc57-7897f36a0d27   154049975                                                                                                                1.5405   500
2015-11-24D05:24:10.716181000   XBUZ15  Buy     10      322.38  MinusTick       215aeaaf-f468-642c-019d-40d823734acb   310192940                                                                                                                3.101929 1000
2015-11-24D06:03:43.821449000   XBUZ15  Sell    10      323.47  PlusTick        e188e3b9-2972-281c-6808-4475db2fc668   309147680                                                                                                                3.091477 1000
2015-11-24D06:05:30.425073000   XBUZ15  Sell    7       323.13  MinusTick       27eaea77-2611-1ae3-3000-4e09b6331a31   216631079                                                                                                                2.166311 700
"""
