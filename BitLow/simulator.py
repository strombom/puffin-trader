import pickle
import random
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta, timezone

from BinanceSimulator.binance_simulator import BinanceSimulator


if 0:
    from scipy.ndimage.filters import gaussian_filter
    with open('cache/tmp.pickle', 'rb') as f:
        data = pickle.load(f)

    #values = data['plots']['tp1.05 sl0.85']
    #x, y = values['x'], values['y']

    #plt.plot(x, y)
    #plt.yscale('log')
    #plt.show()

    drawdowns = {}
    for plot in data['plots']:
        ys = data['plots'][plot]['y']
        max_y = 0
        max_drawdown = 0
        for y in ys:
            max_y = max(max_y, y)
            max_drawdown = max(max_drawdown, (max_y - y) / max_y)
        drawdowns[plot] = max_drawdown

    tps = data['take_profits']
    sls = data['stop_losses']
    plots = data['plots']

    # Make data.
    X = np.array(sls)
    Y = np.array(tps)
    X, Y = np.meshgrid(X, Y)

    Z_equity = np.empty_like(X)
    Z_drawdown = np.empty_like(X)
    for y in range(X.shape[0]):
        for x in range(X.shape[1]):
            idx = int(y * X.shape[1] + x)
            key = list(plots.keys())[idx]
            value = plots[key]['y'][-1] / plots[key]['y'][0]
            # value = np.log10(value)
            # value = max(value, 0.5)
            Z_equity[y, x] = value
            Z_drawdown[y, x] = drawdowns[key]

    Z_equity = gaussian_filter(Z_equity, sigma=1.1)
    #Z_drawdown = gaussian_filter(Z_drawdown, sigma=1.1)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z_drawdown * 10, cmap='Reds')
    ax.plot_surface(X, Y, Z_equity, cmap='PiYG')

    plt.show()

    quit()


class Logger:
    def __init__(self):
        self.timestamps = []
        self.equities = []
        self.plots = {}
        self.positions = {}

    def append(self, timestamp, equity):
        self.timestamps.append(timestamp)
        self.equities.append(equity)

    def save_as(self, name):
        if len(self.equities) > 0:
            print(f"Save {name} {self.equities[-1]}")
        else:
            print(f"Save {name} Empty")
        self.plots[name] = {
            'x': self.timestamps,
            'y': self.equities
        }
        self.timestamps = []
        self.equities = []

    def save_file(self, take_profits, stop_losses):
        with open('cache/tmp.pickle', 'wb') as f:
            pickle.dump({
                'plots': self.plots,
                'take_profits': take_profits,
                'stop_losses': stop_losses
            }, f)

    def plot(self):
        for tp in range(5):
            for sl in range(5):
                idx = tp * 5 + sl
                key = list(self.plots.keys())[idx]
                x = self.plots[key]['x']
                y = self.plots[key]['y']
                plt.plot(x, y, label=key)
        plt.legend()
        plt.show()

    def end_position(self, symbol, kline_idx):
        self.positions[symbol][-1]['kline_end'] = kline_idx

    def start_position(self, symbol, kline_idx):
        if symbol not in self.positions:
            self.positions[symbol] = []
        self.positions[symbol].append({'symbol': symbol, 'kline_start': kline_idx})


class Portfolio:
    def __init__(self):
        pass


def main():
    with open(f"cache/filtered_symbols.pickle", 'rb') as f:
        symbols = pickle.load(f)

    indicators = {}
    for symbol in symbols:
        with open(f"cache/indicators/{symbol}.pickle", 'rb') as f:
            indicators[symbol] = pickle.load(f)

    with open(f"cache/optim_deltas.pickle", 'rb') as f:
        optim_deltas = pickle.load(f)

    data_length = indicators[list(indicators.keys())[0]]['prices'].shape[0]

    """
    datas = {}
    for symbol in optim_deltas[next(iter(optim_deltas))]:
        datas[symbol] = []

    for date in optim_deltas:
        for symbol in datas:
            datas[symbol].append(optim_deltas[date][symbol])

    for symbol in datas:
        plt.plot(datas[symbol], label=symbol)

    plt.legend()
    plt.show()
    quit()
    """

    start_timestamp = datetime.strptime("2020-02-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    indicator_threshold_observe = -0.0011
    indicator_threshold_down = -0.0011
    indicator_threshold_up = 0.0011

    #plt_dat = data['indicators'][12][10]
    #plt.hist(plt_dat, bins=150)
    #plt.show()

    simulator = BinanceSimulator(initial_usdt=1000, symbols=symbols)

    portfolio_size = 10
    portfolio = {}

    logger = Logger()

    def print_hodlings2(kline_idx_):
        timestamp = start_timestamp + timedelta(minutes=kline_idx_)
        total_equity_ = simulator.get_value_usdt()
        stri = f"{timestamp} Hodlings {total_equity_:.1f} USDT"
        for w_symbol in simulator.wallet:
            if simulator.wallet[w_symbol] > 0:
                s_value = simulator.wallet[w_symbol] * simulator.mark_price[w_symbol]
                stri += f", {s_value:.1f} {w_symbol}"
        # print(stri)
        logger.append(timestamp, total_equity_)

    take_profits = np.arange(start=1.01, stop=1.30, step=0.01)
    stop_losses = np.arange(start=0.5, stop=0.995, step=0.02)
    #take_profits = np.arange(start=1.01, stop=1.30, step=0.02)
    #stop_losses = np.arange(start=0.5, stop=0.995, step=0.04)
    #take_profits = np.arange(start=2.75, stop=3.75, step=0.1)
    #stop_losses = np.arange(start=0.825, stop=0.925, step=0.02)
    take_profits = np.arange(start=1.01, stop=1.50, step=0.05)
    stop_losses = np.arange(start=0.70, stop=0.90, step=0.025)
    take_profits = [1.20]
    stop_losses = [0.90]
    print("tp", take_profits)
    print("sl", stop_losses)
    for take_profit in take_profits:
        for stop_loss in stop_losses:
            simulator = BinanceSimulator(initial_usdt=1000, symbols=symbols)

            for kline_idx in range(1, data_length):
                portfolio_symbols = list(portfolio.keys())
                for symbol in portfolio_symbols:
                    mark_price = indicators[symbol]['prices'][kline_idx]
                    simulator.set_mark_price(symbol=symbol, mark_price=mark_price)

                    if mark_price > portfolio[symbol]['take_profit'] or mark_price < portfolio[symbol]['stop_loss']:
                        order_size = -simulator.wallet[symbol]
                        if simulator.market_order(order_size=order_size, symbol=symbol):
                            logger.end_position(symbol, kline_idx)
                            del portfolio[symbol]

                bought = False
                if len(portfolio) < portfolio_size:
                    directions = {}
                    for symbol in symbols:
                        if symbol in portfolio:
                            continue
                        direction_prev = indicators[symbol]['indicators'][0, kline_idx - 1]
                        direction_cur = indicators[symbol]['indicators'][0, kline_idx]
                        if direction_cur < -0.0012 and direction_prev > direction_cur:
                            directions[symbol] = direction_cur

                    if len(directions) > 1:
                        directions = dict(sorted(directions.items(), key=lambda item: item[1]))
                        symbol = list(directions.keys())[-1]
                        mark_price = indicators[symbol]['prices'][kline_idx]
                        simulator.set_mark_price(symbol=symbol, mark_price=mark_price)

                        max_purchase_value = simulator.get_value_usdt() / portfolio_size
                        max_purchase_value = min(max_purchase_value, simulator.wallet['usdt'])
                        order_size = max_purchase_value / mark_price * 0.97

                        if simulator.market_order(order_size=order_size, symbol=symbol):
                            logger.start_position(symbol, kline_idx)

                            timestamp = start_timestamp + timedelta(minutes=kline_idx)
                            timestamp = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                            delta = optim_deltas[timestamp][symbol]
                            diff = 0.95 + delta * 100
                            portfolio[symbol] = {
                                'take_profit': mark_price * (1 + (take_profit - 1) * diff),
                                'stop_loss':  mark_price * (1 - (1 - stop_loss) * diff)
                            }
                            bought = True

                if bought:
                    print_hodlings2(kline_idx)

            print_hodlings2(kline_idx - 1)
            logger.save_as(f"tp{take_profit} sl{stop_loss}")

            symbol = 'ADAUSDT'

            ax1 = plt.subplot(211)
            ax1.plot(indicators[symbol]['prices'])
            for position in logger.positions[symbol]:
                kline_start = position['kline_start']
                if 'kline_end' in position:
                    kline_end = position['kline_end']
                else:
                    kline_end = data_length - 1
                xs = np.arange(kline_start, kline_end)
                ys = indicators[symbol]['prices'][kline_start:kline_end]
                ax1.plot(xs, ys)
            ax1.set_yscale('log')
            plt.setp(ax1.get_xticklabels(), visible=False)

            #xs = np.arange(kline_start, kline_end)
            #ys = indicators[symbol]['indicators'][kline_start:kline_end]
            ax2 = plt.subplot(212, sharex=ax1)
            ax2.plot(indicators[symbol]['indicators'][0])
            plt.show()

            quit()





    logger.save_file(take_profits, stop_losses)


"""
    def print_hodlings(kline_idx_):
        timestamp = start_timestamp + timedelta(minutes=kline_idx_)
        total_equity_ = simulator.get_value_usdt()
        stri = f"{timestamp} Hodlings {total_equity_} USDT"
        for symbol_ in portfolio:
            if portfolio[symbol_]['state'] == 'hodl':
                stri += f", {portfolio[symbol_]['quantity']} {symbol_}"
        print(stri)

    hodling = False

    for kline_idx in range(data['prices'].shape[1]):
        for symbol_idx, symbol in enumerate(data['symbols']):
            simulator.set_mark_price(symbol=symbol, mark_price=data['prices'][symbol_idx][kline_idx])

        symbols_of_interest = []
        for symbol_idx, symbol in enumerate(data['symbols']):
            val = data['indicators'][symbol_idx, 11, kline_idx]
            if val < indicator_threshold_observe:
                symbols_of_interest.append(symbol)

        # Add new symbols to portfolio
        for symbol in symbols_of_interest:
            if symbol not in portfolio or portfolio[symbol]['state'] == 'idle':
                portfolio[symbol] = {
                    'state': 'wait_down'
                }

        # Wait for down movement
        for symbol in portfolio:
            if portfolio[symbol]['state'] == 'wait_down':
                symbol_idx = data['symbols'].index(symbol)
                if data['indicators'][symbol_idx, 5, kline_idx] < indicator_threshold_down:
                    portfolio[symbol]['state'] = 'wait_up'

        # Remove unused symbols from portfolio
        for symbol in portfolio:
            if symbol not in symbols_of_interest:
                if portfolio[symbol]['state'] == ['wait_down', 'wait_up']:
                    portfolio[symbol]['state'] = 'idle'

        # Order
        for symbol in portfolio:
            if not hodling and portfolio[symbol]['state'] == 'wait_up':
                symbol_idx = data['symbols'].index(symbol)
                if data['indicators'][symbol_idx, 4, kline_idx] > indicator_threshold_up:
                    total_equity = simulator.get_value_usdt()
                    volume = total_equity / data['prices'][symbol_idx][kline_idx]
                    max_volume = simulator.wallet['usdt'] / data['prices'][symbol_idx][kline_idx]
                    order_size = min(volume, max_volume) * 0.95
                    simulator.market_order(order_size=order_size, symbol=symbol)
                    portfolio[symbol]['state'] = 'hodl'
                    portfolio[symbol]['quantity'] = order_size
                    mark_price = data['prices'][symbol_idx][kline_idx]
                    portfolio[symbol]['take_profit'] = mark_price * 1.05
                    portfolio[symbol]['stop_loss'] = mark_price * 0.8
                    hodling = True
                    print_hodlings(kline_idx)

        # Stop-loss / take-profit
        for symbol in portfolio:
            if portfolio[symbol]['state'] == 'hodl':
                symbol_idx = data['symbols'].index(symbol)
                mark_price = data['prices'][symbol_idx][kline_idx]
                if mark_price > portfolio[symbol]['take_profit'] or mark_price < portfolio[symbol]['stop_loss']:
                    order_size = -simulator.wallet[symbol]
                    if simulator.market_order(order_size=order_size, symbol=symbol):
                        portfolio[symbol]['quantity'] = 0.0
                        portfolio[symbol]['state'] = 'idle'
                        hodling = False
                        print_hodlings(kline_idx)
"""

"""
idle
wait_down
wait_up






                if portfolio['state'] == 'idle':
                    symbol = random.choice(data['symbols'])
                    symbol_idx = data['symbols'].index(symbol)
                    mark_price = data['prices'][symbol_idx][kline_idx]
                    simulator.set_mark_price(symbol=symbol, mark_price=mark_price)

                    total_equity = simulator.get_value_usdt()
                    volume = total_equity / mark_price
                    max_volume = simulator.wallet['usdt'] / mark_price
                    order_size = min(volume, max_volume) * 0.95
                    simulator.market_order(order_size=order_size, symbol=symbol)
                    mark_price = data['prices'][symbol_idx][kline_idx]
                    portfolio = {
                        'state': 'hodl',
                        'symbol': symbol,
                        'quantity': volume,
                        'take_profit': mark_price * take_profit,
                        'stop_loss':  mark_price * stop_loss
                    }
                    print_hodlings2(kline_idx)

                else:
                    symbol = portfolio['symbol']
                    symbol_idx = data['symbols'].index(symbol)
                    mark_price = data['prices'][symbol_idx][kline_idx]
                    simulator.set_mark_price(symbol=symbol, mark_price=mark_price)

                    if mark_price > portfolio['take_profit'] or mark_price < portfolio['stop_loss']:
                        order_size = -simulator.wallet[symbol]
                        if simulator.market_order(order_size=order_size, symbol=symbol):
                            portfolio = {
                                'state': 'idle',
                                'symbol': 'none',
                                'quantity': 0,
                                'take_profit': 0,
                                'stop_loss': 0
                            }
                            print_hodlings2(kline_idx)
                            
"""

if __name__ == "__main__":
    main()
