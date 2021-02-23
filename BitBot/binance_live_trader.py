
import zmq
import json
import signal
import msgpack
import threading
import ratelimit
from datetime import datetime, timedelta
from multiprocessing import Pipe


def live_data(data_pipe: Pipe):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://192.168.1.90:31003")

    def get_data(last_timestamp: datetime):
        timestamp_start_str = last_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'
        command = {'command': 'get_ticks',
                   'symbol': 'BTCUSDT',
                   'max_rows': 1000,
                   'timestamp_start': timestamp_start_str}
        # print(command)
        command = json.dumps(command).encode()
        socket.send(command)

        message = socket.recv()
        message = msgpack.unpackb(message)

        for row in message:
            # print(row[0] // 1000)
            timestamp = datetime.utcfromtimestamp(row[0] / 1000)
            price = row[1]
            buy = not row[3]
            # print(timestamp, price, buy)
            data_pipe.send((timestamp, price, buy))
            last_timestamp = timestamp + timedelta(milliseconds=1)

        return last_timestamp


    timestamp_start = datetime.utcnow() - timedelta(minutes=2)
    while True:
        if data_pipe.poll(0.5):
            print("live_data has msg " + data_pipe.recv())
            data_pipe.send({'command': 'quit'})
            break
        # print("t start", timestamp_start)
        timestamp_start = get_data(timestamp_start)
        # print("t new", timestamp_start)


def trader(data_pipe: Pipe):
    @ratelimit.limits(calls=1, period=0.5)
    def print_it(msg):
        print("-> " + msg)

    while True:
        if data_pipe.poll(0.5):
            # print(data_pipe.recv())
            timestamp, price, buy = data_pipe.recv()
            try:
                print_it(f"trader has msg {timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')} {price} {buy}")
            except ratelimit.exception.RateLimitException:
                pass


if __name__ == '__main__':

    live_data_pipe, trader_pipe = Pipe()

    live_data_thread = threading.Thread(target=live_data, args=(live_data_pipe, ))
    live_data_thread.start()

    trader_thread = threading.Thread(target=trader, args=(trader_pipe, ))
    trader_thread.start()

    def signal_handler(signal_, frame):
        live_data_pipe.send({'command': 'quit'})

    signal.signal(signal.SIGINT, signal_handler)

    live_data_thread.join()
    trader_thread.join()
