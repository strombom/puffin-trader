import glob
import pickle
from matplotlib import pyplot as plt
import matplotlib.lines as mlines


def get_symbols_thresholds():
    symbols, thresholds = set(), set()
    for filename in glob.glob('cache/thresholds/*.pickle'):
        symbol, threshold = filename.removesuffix('.pickle').split('_')[1:3]
        symbols.add(symbol)
        thresholds.add(float(threshold))
    return sorted(list(symbols)), sorted(list(thresholds))


def main():

    symbols, thresholds = get_symbols_thresholds()

    limits = {
        "ADAUSDT": 3,
        "BCHUSDT": 2.5,
        "BNBUSDT": 3,
        "BTCUSDT": 1.8,
        "BTTUSDT": 6,
        "CHZUSDT": 3.2,
        "EOSUSDT": 3.5,
        "ETCUSDT": 2.5,
        "ETHUSDT": 2.3,
        "MATICUSDT": 11,
        "LINKUSDT": 2.6,
        "THETAUSDT": 2.9,
        "XLMUSDT": 2.3,
        "XRPUSDT": 2.7
    }

    symbols = ["ADAUSDT", "BNBUSDT", "BTTUSDT", "CHZUSDT", "EOSUSDT", "LINKUSDT", "MATICUSDT", "THETAUSDT", "XLMUSDT", "XRPUSDT"]
    #symbols = ["MATICUSDT"]
    #symbols = symbols[7:]

    thresholds = thresholds[2:7]

    print(symbols)
    print(thresholds)

    cols, rows = len(thresholds), len(symbols)
    axs = plt.figure(constrained_layout=True).subplots(rows, cols)
    if cols == 1 and rows == 1:
        axs = [[axs]]
    for row_idx, symbol in enumerate(symbols):
        axs[row_idx][0].set_ylabel(symbol)

    for col_idx, threshold in enumerate(thresholds):
        axs[0][col_idx].set_title(threshold)

    x_values, y_values = {}, {}

    for row_idx, symbol in enumerate(symbols):
        x_values[symbol], y_values[symbol] = {}, {}

        for col_idx, threshold in enumerate(thresholds):
            x_values[symbol][threshold], y_values[symbol][threshold] = [], []

            #print(symbol, threshold)
            try:
                with open(f'cache/thresholds/positions_{symbol}_{threshold:.3f}.pickle', 'rb') as f:
                    positions_data = pickle.load(f)
                    positions = positions_data['positions']
            except FileNotFoundError:
                print("Load positions fail")
                quit()

            dates = []
            all_positions = []
            active_positions = []

            lower_lane = []

            #fee = 0.00075 * 1.25
            fee = -0.0001
            p_up, p_down = 1.010 - fee, 0.990 - fee
            value = 1

            for position in positions:
                active_positions = [active_position for active_position in active_positions if active_position['end'] > position['start']]

                free_lane = 0
                for active_position in active_positions:
                    active_lane = active_position['lane']
                    if active_lane == free_lane:
                        free_lane += 1
                    elif active_lane > free_lane:
                        break

                new_position = {
                    'start': position['start'],
                    'end': position['end'],
                    'gt': position['gt'],
                    'lane': free_lane
                }

                if len(active_positions) == 0:
                    if new_position['gt'] > 0:
                        value *= p_up ** 1
                    else:
                        value *= p_down ** 1
                    x_values[symbol][threshold].append(new_position['end'])
                    y_values[symbol][threshold].append(value)
                    lower_lane.append(new_position)
                    #print(new_position)

                dates.append(position['start'])
                dates.append(position['end'])
                all_positions.append(new_position)
                active_positions.append(new_position)
                active_positions = sorted(active_positions, key=lambda x: x['lane'])

            up, down = 0, 0
            for position in lower_lane:
                if position['gt'] > 0:
                    up += 1
                else:
                    down += 1
                #print(position)

            #print(up, down, up / (up + down))

            #lane_count = max([position['lane'] for position in all_positions]) + 1

            ax = axs[row_idx][col_idx]
            ax.plot(x_values[symbol][threshold], y_values[symbol][threshold])
            #if symbol in limits:
            #    ax.set_ylim(0, limits[symbol])
            #else:
            #    ax.set_ylim(0, 5)

            #dates = sorted(list(set(dates)))
            #fig, ax = plt.subplots()
            #fig.autofmt_xdate()
            #ax.set_xlim(dates[0], dates[-1])
            #plt.plot(x_values, y_values)
            #plt.show()

            """
            for position in all_positions:
                if position['lane'] != 0:
                    continue

                if position['gt'] > 0:
                    color = 'g'
                else:
                    color = 'r'

                ax.barh(
                    y=position['lane'],
                    width=position['end'] - position['start'],
                    left=position['start'],
                    color=color
                )
            """


    plt.show()


if __name__ == '__main__':
    main()
