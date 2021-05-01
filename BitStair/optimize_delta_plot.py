import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize


def curve_fun(x, a, b, c):
    return a / np.log1p(b * x) + c


def curve_fit(x, y):
    popt, _ = scipy.optimize.curve_fit(curve_fun, x, y, p0=[-5.24722065e-01, -9.76472032e-03, -6.61553567e+02])
    return popt


if __name__ == '__main__':
    target = 3000

    deltas = np.array([0.002, 0.004, 0.008, 0.016, 0.032, 0.064])
    pairs = pd.read_csv(filepath_or_buffer='cache/pairs.csv').to_numpy()
    steps = pd.read_csv(filepath_or_buffer='cache/steps.csv').to_numpy()

    deltas2 = np.arange(start=0.002, stop=0.06, step=0.001)

    optim_deltas = {}

    fig, ax = plt.subplots()
    for pair_idx, pair in enumerate(pairs):
        # ax.plot(deltas, steps[pair_idx][1:], label=pair[1])

        popt = curve_fit(deltas, steps[pair_idx][1:])
        res = curve_fun(deltas2, *popt)
        ax.plot(deltas2, res, label=pair[1] + "f")

        res = scipy.optimize.minimize_scalar(lambda x: abs(curve_fun(x, *popt) - target), bounds=(0.005, 0.04), method='bounded')
        optim_delta = res.x
        print(pair[1], optim_delta)
        ax.scatter([optim_delta], [target])

        optim_deltas[pair[1]] = int(optim_delta * 1000) / 1000

    print(optim_deltas)
    with open('cache/optim_deltas.pickle', 'wb') as f:
        pickle.dump(optim_deltas, f, pickle.HIGHEST_PROTOCOL)

    plt.legend()
    plt.show()
