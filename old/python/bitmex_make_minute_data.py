
import sys
import atexit
import signal
import tables
import threading
from pynput import keyboard

from bitmex_raw import BitmexRaw
from bitmex_intervals import BitmexIntervals


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
