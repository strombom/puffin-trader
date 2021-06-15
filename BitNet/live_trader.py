
import zmq
import time
import json
import logging


def main():
    logging.info("Connect to Binance delta server")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:31007")

    command, payload = 'get_since', 1435
    socket.send_pyobj((command, payload))
    message = socket.recv_pyobj()
    print(message['last_idx'])
    print(message['prices'])


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.Formatter.converter = time.gmtime
    main()
