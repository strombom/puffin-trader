
from multiprocessing import Process, Queue


def plot_process(q):
    while True:
        cmd, payload = q.get(block=True)
        if cmd == 'quit':
            break
        elif cmd == 'print':
            print(payload)


class Plot:
    def __init__(self):
        self.q = Queue()
        self.p = Process(target=plot_process, args=(self.q, ))
        self.p.start()

    def shutdown(self):
        self.q.put(('quit', None))
        self.p.join()

    def print(self, msg):
        self.q.put(('print', msg))


if __name__ == '__main__':
    plot = Plot()
    plot.print('abc')
    plot.shutdown()
