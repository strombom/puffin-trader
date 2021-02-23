
import zmq
import json
import msgpack
from datetime import datetime, timedelta


if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://192.168.1.90:31003")

    timestamp_start = datetime.utcnow() - timedelta(minutes=1)
    command = {'command': 'get_ticks',
               'symbol': 'BTCUSDT',
               'max_rows': 5,
               'timestamp_start': timestamp_start.strftime("%Y-%m-%dT%H:%M:%SZ")}
    command = json.dumps(command).encode()
    result = socket.send(command)

    message = socket.recv()
    message = msgpack.unpackb(message)

    for row in message:
        timestamp = row[0]
        price = row[1]
        buy = row[3]
        print(timestamp, price, buy)
