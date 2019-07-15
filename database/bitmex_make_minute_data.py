
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
    vol       = tables.Float32Col(pos =  1)
    price     = tables.Float32Col(pos =  2)
    sell_01   = tables.Float32Col(pos =  3)
    sell_02   = tables.Float32Col(pos =  4)
    sell_05   = tables.Float32Col(pos =  5)
    sell_1    = tables.Float32Col(pos =  6)
    sell_2    = tables.Float32Col(pos =  7)
    sell_5    = tables.Float32Col(pos =  8)
    buy_01    = tables.Float32Col(pos =  9)
    buy_02    = tables.Float32Col(pos = 10)
    buy_05    = tables.Float32Col(pos = 11)
    buy_1     = tables.Float32Col(pos = 12)
    buy_2     = tables.Float32Col(pos = 13)
    buy_5     = tables.Float32Col(pos = 14)


class BitmexRaw:
    def __init__(self):
        self.h5file = tables.open_file("bitmex_raw.h5", mode="r", title="Bitmex")

        try:
            self.xbtusd_ticks_table = self.h5file._get_node('/ticks/XBTUSD')
        except:
            raise RuntimeError('bitmex_raw.h5 does not contain /ticks/XBTUSD')

        self.current_timeperiod = sec15_datetime(self.get_first_timestamp())
        self.row_idx = 0

        self.row_idx = 1747
        self.current_timeperiod = datetime.strptime("2015-10-07 12:45:45", "%Y-%m-%d %H:%M:%S")

        self.a = 0

    def get_first_timestamp(self):
        return datetime.fromtimestamp(self.xbtusd_ticks_table[0][0] / 1000000)

    def get_last_timestamp(self):
        last_idx = self.xbtusd_ticks_table.nrows - 1
        return datetime.fromtimestamp(self.xbtusd_ticks_table[last_idx][0] / 1000000)

    def get_interval_data(self):

        start_idx = self.row_idx
        timeperiod_end = datetime.timestamp(self.current_timeperiod + interval_length) * 1000000

        volume = 0
        count = 0
        while True:
            timestamp = self.xbtusd_ticks_table.cols.timestamp[start_idx + count]
            #print(">>>", timestamp, datetime.fromtimestamp(timestamp / 1000000))
            if timestamp >= timeperiod_end:
                break

            volume += self.xbtusd_ticks_table.cols.volume[start_idx + count]
            count += 1

        self.row_idx += count
        self.current_timeperiod += interval_length

        #print("idx range", start_idx, count)
        #print("tsend", timeperiod_end)

        buys = []
        sells = []
        buy_vol = 0
        sell_vol = 0

        for idx in range(count):
            row = [self.xbtusd_ticks_table.cols.price[start_idx + idx], self.xbtusd_ticks_table.cols.volume[start_idx + idx]]
            vol = self.xbtusd_ticks_table.cols.volume[start_idx + idx]
            if self.xbtusd_ticks_table.cols.buy[start_idx + idx]:
                buys.append(row)
                buy_vol += vol
            else:
                sells.append(row)
                sell_vol += vol

        accum_buy = 0
        accum_sell = 0

        buys  = sorted(buys,  key = itemgetter(0))
        sells = sorted(sells, key = itemgetter(0), reverse = True)

        buy_prices = [0, 0, 0, 0, 0, 0]
        sell_prices = [0, 0, 0, 0, 0, 0]
        steps = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

        #print("sell")
        step_idx = 0
        for sell in sells:
            #print(" ", sell)
            accum_sell += sell[1]
            while step_idx < len(steps) and accum_sell > steps[step_idx]:
                sell_prices[step_idx] = sell[0]
                step_idx += 1
        print("sell prices", sell_prices)

        #print("buy")
        step_idx = 0
        for buy in buys:
            #print(" ", buy)
            accum_buy += buy[1]
            while step_idx < len(steps) and accum_buy > steps[step_idx]:
                buy_prices[step_idx] = buy[0]
                step_idx += 1
        print("buy prices", buy_prices)

        prices = {}

        last_price = self.xbtusd_ticks_table.cols.price[self.row_idx]

        print("row_idx", start_idx - count)
        print("last_price, volume", last_price, volume, buy_vol, sell_vol)
        print("current_timeperiod", self.current_timeperiod - interval_length)
        print("==========")

        #return timestamp, 

        self.a += 1
        if self.a == 1:
            quit()

        #if count == 0:
        #    return



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




bitmex_raw = BitmexRaw()
bitmex_intervals = BitmexIntervals()

"""
start_timeperiod = sec15_datetime(bitmex_raw.get_first_timestamp())
end_timeperiod   = sec15_datetime(bitmex_raw.get_last_timestamp() - timedelta(seconds = 15))

print(start_timeperiod)
print(end_timeperiod)
"""

while True:
    for idx in range(200000):
        interval_data = bitmex_raw.get_interval_data()

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
