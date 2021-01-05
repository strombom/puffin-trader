
from multiprocessing import Process, Pipe
from multiprocessing.connection import Listener
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c


def collect_data(pipe):
    address = 'localhost', 27567
    listener = Listener(address, authkey=b'secret password')

    while True:
        conn = listener.accept()
        print('connection accepted from', listener.last_accepted)
        # while True:
        msg = conn.recv()

        pipe.send(msg)
        #print(ie_scatters)
        #if msg[0] == 'data':
        #    print("rcv data")

        conn.close()


def update_plot(i, pipe, ax1):
    if not pipe.poll():
        return

    msg = pipe.recv()
    ie_scatters, ie_overshoots, trends = msg

    ax1.clear()
    plt.title(f'hello {i}')

    for scatter_idx in range(len(ie_scatters)):
        ie_scatter = ie_scatters[scatter_idx]
        ax1.scatter(ie_scatter['x'], ie_scatter['y'], marker=ie_scatter['marker'], color=ie_scatter['color'],
                    s=ie_scatter['size'])

    ax1.scatter(ie_overshoots['x'], ie_overshoots['y'], marker='_', color='xkcd:grey', s=40)

    """

    #ax1.plot(c_x, c_y, label='C')
    """


if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    recv_pipe, send_pipe = Pipe(False)
    print('Start process...')
    data = Process(target=collect_data, args=(send_pipe,))
    data.start()
    ani = FuncAnimation(fig, update_plot, interval=100, blit=False, fargs=(recv_pipe, ax1))
    plt.show()
    print('...done with process')
    data.join()
    print('Completed multiprocessing')
