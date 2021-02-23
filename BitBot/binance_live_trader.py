
import zmq
import json
import queue
import msgpack
import threading
from time import sleep
from ratelimit import limits
from datetime import datetime, timedelta


def live_data(data_queue: queue.Queue):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://192.168.1.90:31003")

    @limits(calls=2, period=1)
    def get_data(last_timestamp: datetime):
        timestamp_start_str = last_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'
        command = {'command': 'get_ticks',
                   'symbol': 'BTCUSDT',
                   'max_rows': 3,
                   'timestamp_start': timestamp_start_str}
        print(command)
        command = json.dumps(command).encode()
        result = socket.send(command)

        message = socket.recv()
        message = msgpack.unpackb(message)

        for row in message:
            print(row[0] // 1000)
            timestamp = datetime.utcfromtimestamp(row[0] / 1000)
            price = row[1]
            buy = not row[3]
            print(timestamp, price, buy)
            data_queue.put((timestamp, price, buy))
            last_timestamp = timestamp + timedelta(milliseconds=1)

        return last_timestamp

    timestamp_start = datetime.utcnow() - timedelta(minutes=2)
    for i in range(3):
        print("t start", timestamp_start)
        sleep(0.5)
        timestamp_start = get_data(timestamp_start)
        print("t new", timestamp_start)


if __name__ == '__main__':
    live_data_queue = queue.Queue()
    live_data_thread = threading.Thread(target=live_data, args=(live_data_queue, ))
    live_data_thread.start()
    live_data_thread.join()
