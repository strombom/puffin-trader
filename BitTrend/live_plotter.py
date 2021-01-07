
from multiprocessing import Process, Pipe
from multiprocessing.connection import Listener
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
# https://riptutorial.com/matplotlib/example/23558/basic-animation-with-funcanimation

artists = {}

def collect_data(pipe):
    address = 'localhost', 27567
    listener = Listener(address, authkey=b'secret password')

    while True:
        conn = listener.accept()
        print('connection accepted from', listener.last_accepted)
        msg = conn.recv()
        pipe.send(msg)
        conn.close()


def update_plot(i, pipe, ax_1):
    #global artists

    plt.title(f'hello {i}')

    if not pipe.poll():
        return [i for i in list(artists.values()) if i is not None]

    msg = pipe.recv()
    name = msg['name']
    data = msg['data']

    print("recv", name)

    if name not in artists:
        if name == 'ie_scatters':
            for ie_scatter in data:
                ax_1.scatter(ie_scatter['x'],
                             ie_scatter['y'],
                             marker=ie_scatter['marker'],
                             color=ie_scatter['color'],
                             s=ie_scatter['size'])
            artists[name] = None

        elif name == 'ie_overshoots':
            ax_1.scatter(data['x'], data['y'], marker='_', color='xkcd:grey', s=40)
            artists[name] = None

        else:
            artist,  = ax_1.plot(data['x'], data['y'], color=data['color'])
            artist.set_data(data['x'], data['y'])
            artists[name] = artist

    elif name not in ['ie_scatters', 'ie_overshoots']:
        artists[name].set_data(data['x'], data['y'])

    return [i for i in list(artists.values()) if i is not None]


if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    recv_pipe, send_pipe = Pipe(False)
    data_collector = Process(target=collect_data, args=(send_pipe,))
    data_collector.start()
    ani = FuncAnimation(fig, update_plot, interval=100, blit=True, fargs=(recv_pipe, ax1))
    plt.show()
    data_collector.join()
