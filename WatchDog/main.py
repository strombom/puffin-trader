import time
from datetime import datetime, timedelta

from flask import Flask

app = Flask(__name__)

watchdogs = {}


@app.route('/<wdg_name>')
def reset_wdg(wdg_name):
    if wdg_name not in watchdogs:
        watchdogs[wdg_name] = {
            'period': 90,
            'ongoing_alarm': False
        }
    watchdogs['last_reset'] = datetime.now()


@app.route('/<wdg_name>/<timeout_time>')
def setup_wdg(wdg_name, timeout_time):
    if wdg_name not in watchdogs:
        watchdogs[wdg_name] = {
            'last_reset': datetime.now(),
            'period': timedelta(seconds=90),
            'ongoing_alarm': False
        }

    if timeout_time == 'off':
        try:
            del watchdogs[wdg_name]
            return f"Removed {wdg_name}<br><br>{str(watchdogs)}"
        except IndexError:
            return f"Fail"
    try:
        watchdogs[wdg_name]['period'] = timedelta(seconds=int(timeout_time))
        watchdogs[wdg_name]['last_reset'] = datetime.now()
        watchdogs[wdg_name]['ongoing_alarm'] = False
        return f"Setup<br><br>{str(watchdogs)}"
    except ValueError:
        return f"Fail"


def send_email(name):
    print(name)


def wdg_loop():
    while True:
        time.sleep(1)
        for wdg_name in watchdogs:
            watchdog = watchdogs[wdg_name]
            if not watchdog['ongoing_alarm'] and datetime.now() > watchdog['last_reset'] + watchdog['period']:
                watchdog['ongoing_alarm'] = True
                send_email(wdg_name)


if __name__ == '__main__':
    app.run()
