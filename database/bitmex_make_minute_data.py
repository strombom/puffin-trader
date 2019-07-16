
import tables
from datetime import datetime
from datetime import timedelta
from operator import itemgetter

interval_length = timedelta(seconds = 15)

def sec15_datetime(dt):
    second = (dt.second // 15 + 0) * 15
    return dt.replace(second = 0, microsecond = 0)+timedelta(seconds = second)


class IntervalTable(tables.IsDescription):
    timestamp = tables.UInt64Col (pos =  0)
    price     = tables.Float32Col(pos =  1)
    vol_buy   = tables.Float32Col(pos =  2)
    vol_sell  = tables.Float32Col(pos =  3)
    buy_01    = tables.Float32Col(pos =  4)
    buy_02    = tables.Float32Col(pos =  5)
    buy_05    = tables.Float32Col(pos =  6)
    buy_1     = tables.Float32Col(pos =  7)
    buy_2     = tables.Float32Col(pos =  8)
    buy_5     = tables.Float32Col(pos =  9)
    sell_01   = tables.Float32Col(pos = 10)
    sell_02   = tables.Float32Col(pos = 11)
    sell_05   = tables.Float32Col(pos = 12)
    sell_1    = tables.Float32Col(pos = 13)
    sell_2    = tables.Float32Col(pos = 14)
    sell_5    = tables.Float32Col(pos = 15)


class BitmexRaw:
    def __init__(self, bitmex_intervals):
        self.h5file = tables.open_file("bitmex_raw.h5", mode="r", title="Bitmex")

        try:
            self.xbtusd_ticks_table = self.h5file._get_node('/ticks/XBTUSD')
        except:
            raise RuntimeError('bitmex_raw.h5 does not contain /ticks/XBTUSD')

        if start_timeperiod == 0:
            self.current_timeperiod = sec15_datetime(self.get_first_timestamp())
            self.current_row_idx = 0
        else:
            self.current_timeperiod = start_timeperiod
            self.current_row_idx = start_row_idx

        #self.row_idx = 1747
        #self.current_timeperiod = datetime.strptime("2015-10-07 12:45:45", "%Y-%m-%d %H:%M:%S")

    def get_first_timestamp(self):
        return datetime.fromtimestamp(self.xbtusd_ticks_table[0][0] / 1000000)

    def get_last_timestamp(self):
        last_idx = self.xbtusd_ticks_table.nrows - 1
        return datetime.fromtimestamp(self.xbtusd_ticks_table[last_idx][0] / 1000000)

    def get_interval_data(self):
        start_idx = self.current_row_idx
        timeperiod_start = datetime.timestamp(self.current_timeperiod) * 1000000
        timeperiod_end   = datetime.timestamp(self.current_timeperiod + interval_length) * 1000000

        count = 0
        while True:
            timestamp = self.xbtusd_ticks_table.cols.timestamp[start_idx + count]
            if timestamp >= timeperiod_end:
                break
            count += 1

        self.current_row_idx += count
        self.current_timeperiod += interval_length

        buys, sells = [], []
        vol_buy, vol_sell = 0, 0

        for idx in range(count):
            price = self.xbtusd_ticks_table.cols.price [start_idx + idx]
            vol   = self.xbtusd_ticks_table.cols.volume[start_idx + idx]
            row   = [price, vol]
            if self.xbtusd_ticks_table.cols.buy[start_idx + idx]:
                buys.append(row)
                vol_buy += vol
            else:
                sells.append(row)
                vol_sell += vol

        buys  = sorted(buys,  key = itemgetter(0))
        sells = sorted(sells, key = itemgetter(0), reverse = True)

        prices_buy, prices_sell = [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]
        steps = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        accum_buy, accum_sell = 0, 0

        step_idx = 0
        for buy in buys:
            accum_buy += buy[1]
            while step_idx < len(steps) and accum_buy > steps[step_idx]:
                prices_buy[step_idx] = buy[0]
                step_idx += 1

        step_idx = 0
        for sell in sells:
            accum_sell += sell[1]
            while step_idx < len(steps) and accum_sell > steps[step_idx]:
                prices_sell[step_idx] = sell[0]
                step_idx += 1

        prices = {}

        last_price = self.xbtusd_ticks_table.cols.price[self.row_idx]

        return timeperiod_start, last_price, vol_sell, vol_buy, prices_buy, prices_sell

    def get_start_timeperiod_and_start_row_idx(self):
        return start_timeperiod, start_row_idx



class BitmexIntervals:
    def __init__(self):
        self.h5file = tables.open_file("bitmex_intervals.h5", mode="a", title="Bitmex")

        try:
            self.ticks_group = self.h5file.create_group("/", 'sec15', 'Tick data')
        except:
            self.ticks_group = self.h5file._get_node('/sec15')

        try:
            self.xbtusd_15sec_table = self.h5file.create_table('/sec15', 'XBTUSD', IntervalTable)
        except:
            self.xbtusd_15sec_table = self.h5file._get_node('/sec15/XBTUSD')

        try:
            self.start_timeperiod = self.xbtusd_15sec_table.attrs.START_TIMEPERIOD
        except:
            self.start_timeperiod = 0

        try:
            self.start_row_idx = self.xbtusd_15sec_table.attrs.START_ROW_IDX
        except:
            self.start_row_idx = 0

    def save_start_timeperiod_and_row_idx(self, start_timeperiod, start_row_idx):
        self.xbtusd_15sec_table.attrs.START_TIMEPERIOD = start_timeperiod
        self.xbtusd_15sec_table.attrs.START_ROW_IDX    = start_row_idx




bitmex_intervals = BitmexIntervals()
bitmex_raw       = BitmexRaw(bitmex_intervals.start_timeperiod, bitmex_intervals.start_row_idx)

"""
start_timeperiod = sec15_datetime(bitmex_raw.get_first_timestamp())
end_timeperiod   = sec15_datetime(bitmex_raw.get_last_timestamp() - timedelta(seconds = 15))

print(start_timeperiod)
print(end_timeperiod)
"""

while True:
    for idx in range(200000):
        interval_data = bitmex_raw.get_interval_data()

        timestamp, last_price, vol_sell, vol_buy, prices_buy, prices_sell = interval_data

        start_timeperiod, start_row_idx = bitmex_raw.get_start_timeperiod_and_start_row_idx()
        bitmex_intervals.save_start_timeperiod_and_row_idx(start_timeperiod, start_row_idx)

        print("interval", timestamp, last_price, vol_sell, vol_buy, prices_buy, prices_sell)
        quit()

        print("row_idx", start_idx - count)
        print("last_price, volume", last_price, vol_buy + vol_sell, vol_buy, vol_sell)
        print("current_timeperiod", self.current_timeperiod - interval_length)
        print("==========")


    print("interval")
    print(interval_data)



    quit()


quit()


print("first   timestamp ", datetime.fromtimestamp(xbtusd_ticks_table[0][0] / 1000000))
print("start   timeperiod", start_timeperiod)
print("current timestamp ", datetime.now())
print("end     timeperiod", end_timeperiod)


i = 0
raw_idx = 0
timestamp = start_timestamp
next_timestamp = start_timestamp + timedelta(seconds = 15)
while timestamp < end_timestamp:
    timestamp = datetime.fromtimestamp(xbtusd_ticks_table.cols.timestamp[raw_idx] / 1000000)
    price     = xbtusd_ticks_table.cols.price [raw_idx]
    volume    = xbtusd_ticks_table.cols.volume[raw_idx]

    #while timestamp 



    print("ts", timestamp)

    if timestamp > next_timestamp:
        print("next timestamp!")

        i += 1
        if i > 150:
            quit()



    print("price vol", price, volume)

    raw_idx += 1
    #if raw_idx == 1:
    #    quit()
