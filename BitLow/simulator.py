import pickle
from datetime import datetime, timedelta, timezone

from matplotlib import pyplot as plt

from BinanceSimulator.binance_simulator import BinanceSimulator


def main():

    with open(f"cache/indicators.pickle", 'rb') as f:
        data = pickle.load(f)
        print(data)

    start_timestamp = datetime.strptime("2020-02-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    indicator_threshold_observe = -0.0015
    indicator_threshold_down = -0.001
    indicator_threshold_up = 0.0005

    #plt_dat = data['indicators'][12][10]
    #plt.hist(plt_dat, bins=150)
    #plt.show()

    simulator = BinanceSimulator(initial_usdt=1000, symbols=data['symbols'])
    portfolio = {}

    hodling = False

    def print_hodlings(kline_idx_):
        timestamp = start_timestamp + timedelta(minutes=kline_idx_)
        total_equity_ = simulator.get_value_usdt()
        stri = f"{timestamp} Hodlings {total_equity_} USDT"
        for symbol_ in portfolio:
            if portfolio[symbol_]['state'] == 'hodl':
                stri += f", {portfolio[symbol_]['quantity']} {symbol_}"
        print(stri)

    for kline_idx in range(data['prices'].shape[1]):
        for symbol_idx, symbol in enumerate(data['symbols']):
            simulator.set_mark_price(symbol=symbol, mark_price=data['prices'][symbol_idx][kline_idx])

        symbols_of_interest = []
        for symbol_idx, symbol in enumerate(data['symbols']):
            val = data['indicators'][symbol_idx, 10, kline_idx]
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
                if data['indicators'][symbol_idx, 3, kline_idx] < indicator_threshold_down:
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
                if data['indicators'][symbol_idx, 1, kline_idx] > indicator_threshold_up:
                    total_equity = simulator.get_value_usdt()
                    volume = total_equity / data['prices'][symbol_idx][kline_idx]
                    max_volume = simulator.wallet['usdt'] / data['prices'][symbol_idx][kline_idx]
                    order_size = min(volume, max_volume) * 0.95
                    simulator.market_order(order_size=order_size, symbol=symbol)
                    portfolio[symbol]['state'] = 'hodl'
                    portfolio[symbol]['quantity'] = order_size
                    mark_price = data['prices'][symbol_idx][kline_idx]
                    portfolio[symbol]['take_profit'] = mark_price * 1.015
                    portfolio[symbol]['stop_loss'] = mark_price * 0.92
                    hodling = True
                    print_hodlings(kline_idx)

        # Stop-loss / take-profit
        for symbol in portfolio:
            if portfolio[symbol]['state'] == 'hodl':
                symbol_idx = data['symbols'].index(symbol)
                mark_price = data['prices'][symbol_idx][kline_idx]
                if mark_price > portfolio[symbol]['take_profit'] or mark_price < portfolio[symbol]['stop_loss']:
                    order_size = -portfolio[symbol]['quantity']
                    if simulator.market_order(order_size=order_size, symbol=symbol):
                        portfolio[symbol]['quantity'] = 0.0
                        portfolio[symbol]['state'] = 'idle'
                        hodling = False
                        print_hodlings(kline_idx)

"""
idle
wait_down
wait_up
"""

if __name__ == "__main__":
    main()
