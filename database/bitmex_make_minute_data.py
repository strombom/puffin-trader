
import sys
import atexit
import signal
import tables
import threading
from pynput import keyboard
from datetime import datetime
from datetime import timedelta
from operator import itemgetter

interval_length = timedelta(seconds = 15)

def sec15_datetime(dt):
    second = (dt.second // 15 + 0) * 15
    return dt.replace(second = 0, microsecond = 0)+timedelta(seconds = second)


class IntervalTable(tables.IsDescription):
    timestamp = tables.UInt64Col (pos =  0) # Microseconds
    price     = tables.Float32Col(pos =  1) # USD
    vol_buy   = tables.Float32Col(pos =  2) # BTC
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

        self.bitmex_intervals.raw_start_timeperiod = datetime.timestamp(self.current_timeperiod) * 1e6
        self.bitmex_intervals.raw_start_row_idx    = self.current_row_idx

        last_price = self.xbtusd_ticks_table.cols.price[self.current_row_idx - 1]
        return timeperiod_start, last_price, vol_sell, vol_buy, prices_buy, prices_sell


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

        self.load_start_timeperiod_and_row_idx()
    
    def load_start_timeperiod_and_row_idx(self):
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
        for idx, key in enumerate(['01', '02', '05', '1', '2', '5']):
            self.xbtusd_15sec_table.row['buy_'  + key] = prices_buy[idx]
            self.xbtusd_15sec_table.row['sell_' + key] = prices_sell[idx]
        self.xbtusd_15sec_table.row.append()

    def flush(self):
        self.save_start_timeperiod_and_row_idx(force = True)
        self.xbtusd_15sec_table.flush()

running = True

def minute_data_loop():
    global running
    bitmex_intervals = BitmexIntervals()
    bitmex_raw       = BitmexRaw(bitmex_intervals)
    count = 0
    while running:
        try:
            timestamp, last_price, vol_sell, vol_buy, prices_buy, prices_sell = bitmex_raw.get_interval_data()
            bitmex_intervals.append(timestamp, last_price, vol_sell, vol_buy, prices_buy, prices_sell)
            
            count += 1
            if count == 10000:
                count = 0
                bitmex_intervals.flush()
                print(timestamp, last_price, vol_sell, vol_buy, prices_buy, prices_sell)
                print("start", bitmex_intervals.raw_start_timeperiod, bitmex_intervals.raw_start_row_idx)
                print("===")
        
        except IndexError:
            bitmex_intervals.flush()
            break

    bitmex_intervals.flush()
    print("minute data loop end")

def save_on_key_exit(key):
    #print('{0} press'.format(key))
    if key == keyboard.Key.esc:
        print('Exit key, saving.')
        global running
        running = False

def key_listener_loop():
    global running
    with keyboard.Listener(on_press=save_on_key_exit) as listener:
        listener.join()
    print("key listener loop exit")

minute_data_thread  = threading.Thread(target=minute_data_loop)
key_listener_thread = threading.Thread(target=key_listener_loop)
key_listener_thread.daemon = True


"""
def signal_handler(sig, frame):
    print('Exit signal, ctrl+c.')
    #sys.exit(0)
    global running
    running = False
    #minute_data_thread.join()

def save_on_exit():
    print('Exit, saving.')
    global running
    running = False
    #bitmex_intervals.flush()
    minute_data_thread.join()
    quit()

signal.signal(signal.SIGINT, signal_handler)
atexit.register(save_on_exit)
"""

minute_data_thread.start()
key_listener_thread.start()
minute_data_thread.join()
