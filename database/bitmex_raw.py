
import tables
from operator import itemgetter
from datetime import datetime
from datetime import timedelta


interval_length = timedelta(seconds = 15)

def sec15_datetime(dt):
    second = (dt.second // 15 + 0) * 15
    return dt.replace(second = 0, microsecond = 0) + timedelta(seconds = second)

class BitmexRaw:
    def __init__(self, bitmex_intervals):
        self.bitmex_intervals = bitmex_intervals
        self.h5file = tables.open_file("bitmex_raw.h5", mode="r", title="Bitmex")

        try:
            self.xbtusd_ticks_table = self.h5file._get_node('/ticks/XBTUSD')
        except:
            raise RuntimeError('bitmex_raw.h5 does not contain /ticks/XBTUSD')

        if bitmex_intervals.raw_start_timeperiod == 0:
            self.current_timeperiod = sec15_datetime(self.get_first_timestamp())
            self.current_row_idx = 0
        else:
            timeperiod = self.bitmex_intervals.raw_start_timeperiod
            timeperiod = datetime.fromtimestamp(timeperiod / 1e6)
            self.current_timeperiod = sec15_datetime(timeperiod)
            self.current_row_idx = self.bitmex_intervals.raw_start_row_idx

    def get_first_timestamp(self):
        return datetime.fromtimestamp(self.xbtusd_ticks_table[0][0] / 1e6)

    def get_last_timestamp(self):
        last_idx = self.xbtusd_ticks_table.nrows - 1
        return datetime.fromtimestamp(self.xbtusd_ticks_table[last_idx][0] / 1e6)

    def get_interval_data(self):
        if self.current_row_idx >= self.xbtusd_ticks_table.nrows:
            raise IndexError
        
        start_idx = self.current_row_idx
        timeperiod_start = datetime.timestamp(self.current_timeperiod) * 1e6
        timeperiod_end   = datetime.timestamp(self.current_timeperiod + interval_length) * 1e6

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
        steps = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
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

        self.bitmex_intervals.raw_start_timeperiod = datetime.timestamp(self.current_timeperiod) * 1e6
        self.bitmex_intervals.raw_start_row_idx    = self.current_row_idx

        last_price = self.xbtusd_ticks_table.cols.price[self.current_row_idx - 1]
        return timeperiod_start, last_price, vol_sell, vol_buy, prices_buy, prices_sell
