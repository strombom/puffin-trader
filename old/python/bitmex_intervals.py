
import tables


class IntervalTable(tables.IsDescription):
    timestamp = tables.UInt64Col (pos =  0) # Microseconds
    price     = tables.Float32Col(pos =  1) # USD
    vol_buy   = tables.Float32Col(pos =  2) # BTC
    vol_sell  = tables.Float32Col(pos =  3)
    buy_01    = tables.Float32Col(pos =  4)
    buy_02    = tables.Float32Col(pos =  5)
    buy_05    = tables.Float32Col(pos =  6)
    buy_10    = tables.Float32Col(pos =  7)
    buy_20    = tables.Float32Col(pos =  8)
    buy_50    = tables.Float32Col(pos =  9)
    sell_01   = tables.Float32Col(pos = 10)
    sell_02   = tables.Float32Col(pos = 11)
    sell_05   = tables.Float32Col(pos = 12)
    sell_10   = tables.Float32Col(pos = 13)
    sell_20   = tables.Float32Col(pos = 14)
    sell_50   = tables.Float32Col(pos = 15)


class BitmexIntervals:
    raw_start_row_idx = 0
    raw_start_timeperiod = 0

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

        self._load_start_timeperiod_and_row_idx()
    
    def _load_start_timeperiod_and_row_idx(self):
        try:
            self.raw_start_timeperiod = self.xbtusd_15sec_table.attrs.START_TIMEPERIOD
        except:
            self.raw_start_timeperiod = 0
        self.last_saved_timeperiod = self.raw_start_timeperiod

        try:
            self.raw_start_row_idx = self.xbtusd_15sec_table.attrs.START_ROW_IDX
        except:
            self.raw_start_row_idx = 0
        
        print("load", self.raw_start_row_idx, self.raw_start_timeperiod)
    
    def get_start_timeperiod_and_start_row_idx(self):
        return self.raw_start_timeperiod, self.raw_start_row_idx

    def save_start_timeperiod_and_row_idx(self, force = False):
        delta_seconds = self.raw_start_timeperiod - self.last_saved_timeperiod
        if force or delta_seconds > 100:
            print("saving")
            self.xbtusd_15sec_table.attrs.START_TIMEPERIOD = (int) (self.raw_start_timeperiod)
            self.xbtusd_15sec_table.attrs.START_ROW_IDX    = self.raw_start_row_idx
    
    def append(self, timestamp, last_price, vol_sell, vol_buy, prices_buy, prices_sell):
        self.xbtusd_15sec_table.row['timestamp']  = (int) (timestamp)
        self.xbtusd_15sec_table.row['price'] = last_price
        self.xbtusd_15sec_table.row['vol_sell']   = vol_sell
        self.xbtusd_15sec_table.row['vol_buy']    = vol_buy
        for idx, key in enumerate(['01', '02', '05', '10', '20', '50']):
            self.xbtusd_15sec_table.row['buy_'  + key] = prices_buy[idx]
            self.xbtusd_15sec_table.row['sell_' + key] = prices_sell[idx]
        self.xbtusd_15sec_table.row.append()

    def flush(self):
        self.save_start_timeperiod_and_row_idx(force = True)
        self.xbtusd_15sec_table.flush()
