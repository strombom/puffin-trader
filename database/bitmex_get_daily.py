
import io
import csv
import glob
import gzip
import datetime
import urllib.request
import tables

"""
csv_files = []
for file in glob.glob("bitmex_daily/*.csv"):
    csv_files.append(file)

print(csv_files)

#date_first = "20141122"
#date_first = "20141124"
#date_last = "20190626"
url = "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/xxxxxxxx.csv.gz"


date_first = datetime.datetime.strptime("20141122", '%Y%m%d')
date_last  = datetime.datetime.strptime("20151124", '%Y%m%d')



def download_file(date):
    date_string = datetime.datetime.strftime(date, '%Y%m%d')
    date_url = url.replace("xxxxxxxx", date_string)
    url_request = urllib.request.Request(date_url)
    url_connect = urllib.request.urlopen(url_request)
    data_gzip = url_connect.read(10**10)
    data_raw = gzip.GzipFile(fileobj=io.BytesIO(data_gzip)).read().decode("utf-8") 
    reader = csv.reader(data_raw.split('\n'), delimiter=',')

    rows = []
    for row in reader:
        print('\t'.join(row))
        rows.append(row)

    import pickle
    with open('trade_data.pickle', 'wb') as f:
        pickle.dump(rows, f, pickle.HIGHEST_PROTOCOL)


download_file(date_last)

quit()
"""


class TickTable(tables.IsDescription):
    timestamp   = tables.UInt64Col()
    price       = tables.Float32Col()
    volume      = tables.Float32Col()

class TradeTable(tables.IsDescription):
    timestamp   = tables.UInt64Col()
    price_high  = tables.Float32Col()
    price_low   = tables.Float32Col()
    volume      = tables.Float32Col()

class SymbolTable(tables.IsDescription):
    name        = tables.StringCol(16)
    ts_start    = tables.Float32Col()
    ts_stop     = tables.Float32Col()



h5file = tables.open_file("bitmex.h5", mode="a", title="Bitmex")
print("open", h5file)

try:
    ticks_group = h5file.create_group("/", 'ticks', 'Tick data')
except:
    ticks_group = h5file._get_node('/ticks')

try:
    symbols_group = h5file.create_group("/", 'symbols', 'Symbol data')
except:
    symbols_group = h5file._get_node('/symbols')



symbols = []

import pickle
with open('trade_data.pickle', 'rb') as f:
    trade_data = pickle.load(f)

for row in trade_data[1:]:
    if not row:
        break
    symbol = row[1]
    if symbol not in symbols:
        print(row)
        print("s", symbol)
        quit()
quit()




for symbol in symbols_group:
    print("symbol", symbol)

tables = {}

for node in trades_group:
    print("node", node)

quit()


try:
    trades_table = h5file.create_table(trades_group, 'XBUZ15', TradeTable, "Trade")
except:
    trades_table = h5file._get_node('/ticks/XBUZ15')

trade = trades_table.row

trade['timestamp'] = 124
trade['price_high'] = 101.5
trade['price_low'] = 101.2
trade['volume'] = 3.0
trade.append()

trades_table.flush()


print(h5file)

print(trades_table)


"""
2015-11-24D17:19:40.331762000   XBTZ15  Buy     1       342     ZeroPlusTick    d3253a44-71bb-010d-b383-e60b6b4dac48   342000                                                                                                                   0.00342  1.16964
2015-11-24D01:13:28.846232000   XBUZ15  Sell    5       324.57  PlusTick        1e619051-6aba-06ef-cc57-7897f36a0d27   154049975                                                                                                                1.5405   500
2015-11-24D05:24:10.716181000   XBUZ15  Buy     10      322.38  MinusTick       215aeaaf-f468-642c-019d-40d823734acb   310192940                                                                                                                3.101929 1000
2015-11-24D06:03:43.821449000   XBUZ15  Sell    10      323.47  PlusTick        e188e3b9-2972-281c-6808-4475db2fc668   309147680                                                                                                                3.091477 1000
2015-11-24D06:05:30.425073000   XBUZ15  Sell    7       323.13  MinusTick       27eaea77-2611-1ae3-3000-4e09b6331a31   216631079                                                                                                                2.166311 700
"""


