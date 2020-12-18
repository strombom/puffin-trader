
import matplotlib.pyplot as plt
from ratelimit import limits, sleep_and_retry
from multiprocessing import Process, Pipe


def plot_process(conn):
    @sleep_and_retry
    @limits(calls=1, period=0.1)
    def on_mouse_move(event):
        conn.send(('x', event.xdata))

    #fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all')
    fig, ax1 = plt.subplots(1, 1, sharex='all')
    fig.tight_layout()
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    while True:
        cmd, payload = conn.recv()

        if cmd == 'quit':
            break

        elif cmd == 'plot':
            #x, ie_prices, target_direction, measured_direction = payload
            x, ie_prices = payload
            ax1.plot(x, ie_prices)
            #for idx in range(target_direction.shape[0]):
            #    ax2.fill_between(x, 1 * idx, 1 * idx + target_direction[idx])
            #for idx in range(measured_direction.shape[0]):
            #    ax3.fill_between(x, 1 * idx, 1 * idx + measured_direction[idx])

        elif cmd == 'show':
            plt.show()
            conn.send(('quit', None))

        elif cmd == 'print':
            print(payload)


class Plot:
    def __init__(self):
        self.conn, conn_remote = Pipe()
        self.p = Process(target=plot_process, args=(conn_remote, ))
        self.p.start()

    def shutdown(self):
        self.conn.send(('quit', None))
        self.p.join()

    def get(self):
        return self.conn.recv()

    def print(self, msg):
        self.conn.send(('print', msg))

    def plot(self, payload):
        self.conn.send(('plot', payload))

    def show(self):
        self.conn.send(('show', None))


if __name__ == '__main__':
    plot = Plot()
    plot.plot('abc')

    print('a')
    while True:
        cmd, payload = plot.get()
        if cmd == 'quit':
            break
        elif cmd == 'x':
            print("got x", payload)

    plot.print('abc')
    plot.shutdown()
