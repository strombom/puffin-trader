
import sys
sys.path.append("../Common")

from Common.misc import read_events, string_to_timestamp


settings = {'events_filepath': '../../tmp/PD_Events/events.csv',
            'data_first_timestamp': string_to_timestamp("2020-01-01 00:00:00.000"),
            'start_timestamp': string_to_timestamp("2020-01-01 00:00:00.000"),
            'end_timestamp': string_to_timestamp("2020-02-01 00:00:00.000")
            }

events = read_events(settings)


