
import tables
from datetime import datetime
from datetime import timedelta

h5file = tables.open_file("bitmex_test.h5", mode="a", title="Bitmex")

try:
    symbols_table = h5file._get_node('/symbols')
except:
    print("Bad file")
    quit()

try:
    ticks_group = h5file._get_node('/ticks')
except:
    print("Bad file")
    quit()

try:
    xbtusd_table = h5file._get_node('/ticks/XBTUSD')
except:
    print("Bad file")
    quit()

print(xbtusd_table[0])


def quarter_minute_datetime(dt):
    second = (dt.second//15+0)*15
    return dt.replace(second=0, microsecond=0)+timedelta(seconds=second)

first_timestamp = quarter_minute_datetime(datetime.fromtimestamp(xbtusd_table[0][0] / 1000000))

print(first_timestamp)


