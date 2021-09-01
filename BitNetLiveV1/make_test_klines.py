import pandas as pd
import zmq


def make_test_klines():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://192.168.1.90:31007")

    command, payload = 'get_since', 0
    socket.send_pyobj((command, payload))
    message = socket.recv_pyobj()

    df = pd.DataFrame(data=message['prices']).iloc[-10000:]
    df.to_csv(path_or_buf='C:/BitBotLiveV1/test_klines.csv')

    print(message)


if __name__ == '__main__':
    make_test_klines()
