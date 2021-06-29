import json
import time
import pickle
import smtplib
import threading
import urllib.parse
from flask import Flask
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

app = Flask(__name__)

watchdogs = {}


def save_watchdogs():
    with open(f"wdg.pickle", 'wb') as f:
        pickle.dump(watchdogs, f, pickle.HIGHEST_PROTOCOL)


def load_watchdogs():
    global watchdogs
    try:
        with open(f"wdg.pickle", 'rb') as f:
            tmp_watchdogs = pickle.load(f)
            for wdg_name in tmp_watchdogs:
                tmp_watchdogs[wdg_name]['last_reset'] = datetime.now()
            watchdogs = tmp_watchdogs

    except FileNotFoundError:
        pass


@app.route('/<wdg_name>')
def reset_wdg(wdg_name):
    if wdg_name not in watchdogs:
        watchdogs[wdg_name] = {
            'period': timedelta(seconds=90),
            'ongoing_alarm': False
        }
    watchdogs[wdg_name]['last_reset'] = datetime.now()
    save_watchdogs()
    return f"Reset wdg_name<br><br>{watchdogs}"


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


@app.route('/status')
def status_wdg():
    return f"Status<br><br>{watchdogs}"


def send_email(name):
    subject = f"BitBot Watchdog Alert {name}"
    message = f"BitBot Watchdog Alert {name}"

    with open('mail_account.json') as f:
        account_info = json.load(f)

    port_number = account_info['account_port']
    msg = MIMEMultipart()
    msg['From'] = account_info['email_from']
    msg['To'] = account_info['email_to']
    msg['Subject'] = subject
    msg.attach(MIMEText(message))
    mailserver = smtplib.SMTP('localhost', port_number)
    mailserver.login(account_info['account_email'], account_info['account_password'])
    mailserver.sendmail(account_info['email_from'], account_info['email_to'], msg.as_string())
    mailserver.quit()


def wdg_loop():
    while True:
        time.sleep(1)
        for wdg_name in watchdogs:
            watchdog = watchdogs[wdg_name]
            if not watchdog['ongoing_alarm'] and datetime.now() > watchdog['last_reset'] + watchdog['period']:
                watchdog['ongoing_alarm'] = True
                send_email(wdg_name)
                print(f"Watchdog alarm {wdg_name}")


if __name__ == '__main__':
    load_watchdogs()
    x = threading.Thread(target=wdg_loop)
    x.start()
    app.run(host='localhost', port=31008)
