import collections
import itertools

import zmq
import time
import logging
import numpy as np
import pandas as pd
from fastai.learner import load_learner

from IntrinsicTime.runner import Runner


def main():
    profit_model = load_learner('model_all.pickle')

    logging.info("Connect to Binance delta server")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:31007")

    delta = 0.01
    direction_degrees = [1, 2, 3]
    lengths = pd.read_csv('cache/regime_data_lengths.csv')['length'].to_list()
    step_count = lengths[-1]

    # indicators = None
    last_data_idx = 0
    symbols = []
    steps = {}
    runners = {}

    directions = None

    directions_column_names = []
    for _, direction_degree in enumerate(direction_degrees):
        for _, length in enumerate(lengths):
            directions_column_names.append(f"{direction_degree}-{length}")

    while True:
        command, payload = 'get_since', last_data_idx
        socket.send_pyobj((command, payload))
        message = socket.recv_pyobj()
        last_data_idx = message['last_idx']
        prices = message['prices']

        if len(symbols) == 0:
            symbols = sorted(prices[0].keys())
            directions = np.empty((len(symbols), len(direction_degrees) * len(lengths)))

        # Runners
        for price in prices:
            for symbol in symbols:
                if symbol not in runners:
                    runners[symbol] = Runner(delta=delta)
                    steps[symbol] = collections.deque(maxlen=step_count)
                runner_steps = runners[symbol].step(price[symbol])
                for step in runner_steps:
                    steps[symbol].append(step)
                    # print(symbol, step)

        # Make directions
        for symbol_idx, symbol in enumerate(symbols):
            for direction_degree_idx, direction_degree in enumerate(direction_degrees):
                for length_idx, length in enumerate(lengths):
                    idx = lengths[-1]
                    start, end = idx - length, idx
                    xp = np.arange(start, end)
                    direction_steps = list(itertools.islice(steps[symbol], start, end))
                    yp = np.poly1d(np.polyfit(xp, direction_steps, direction_degree))
                    curve = yp(xp)
                    direction = curve[-1] / curve[-2] - 1.0
                    directions[symbol_idx, direction_degree_idx * len(lengths) + length_idx] = direction

        # Predict values
        data_input = pd.DataFrame(data=directions, columns=directions_column_names)
        input_symbols = np.array(symbols)
        for symbol in symbols:
            data_input[symbol] = np.where(input_symbols == symbol, True, False)
        test_dl = profit_model.dls.test_dl(data_input)
        predictions = profit_model.get_preds(dl=test_dl)[0][:, 2].numpy() - 0.5

        quit()


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.Formatter.converter = time.gmtime
    main()
