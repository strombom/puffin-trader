

class ByBitSimulator:
    def __init__(self, initial_usdt, symbols):
        #self.liquidation_level = 1.1
        self.symbols = symbols
        self.positions_buy = {}
        self.positions_sell = {}
        self.wallet_usdt = initial_usdt
        self.wallet = {}
        self.mark_price = {}
        for symbol in self.symbols:
            self.wallet[symbol] = 0.0
            self.positions_buy[symbol] = 0.0
            self.positions_sell[symbol] = 0.0
            self.mark_price[symbol] = 0.0

    #def get_positions_symbols(self):
    #    symbols = set()
    #    for symbol in self.wallet:
    #        if symbol != 'usdt' and self.wallet[symbol] > 0:
    #            symbols.add(symbol)
    #    return symbols

    def is_liquidated(self):
        if self.wallet['BTCUSDT'] < 0.001 and self.wallet_usdt < 10:
            return True
        return False

    #def is_liquidated(self):
    #    return False
    #    # if self.wallet_btc == 0 and self.wallet_usdt == 0:
    #    #    return True
    #    # else:
    #    #    return False

    #def calculate_leverage(self):
    #    if self.is_liquidated():
    #        return -1

    #    total_asset_value, total_debt = 0.0, 0.0
    #    for pair in self.wallet:
    #        total_asset_value += self.wallet[pair] * self.mark_price[pair]
    #        total_debt += self.debt[pair] * self.mark_price[pair]
    #    total_equity = total_asset_value - total_debt

    #    if total_debt >= total_asset_value:
    #        for pair in self.wallet:
    #            self.wallet[pair] = 0.0
    #            self.debt[pair] = 0.0
    #        return -1

    #    return total_asset_value / total_equity - 1

    def set_mark_price(self, symbol, mark_price):
        self.mark_price[symbol] = mark_price

    def get_equity_usdt(self):
        value = self.wallet_usdt
        for pair in self.wallet:
            value += self.wallet[pair] * self.mark_price[pair]
            #value += (self.wallet[pair] - self.debt[pair]) * self.mark_price[pair]
        return value

    def get_cash_usdt(self):
        return self.wallet_usdt  # - self.debt['usdt']

    def calculate_margin(self):
        total_asset_value = self.wallet['usdt']
        total_debt = self.debt['usdt']
        for pair in self.symbols:
            total_asset_value += self.wallet[pair] * self.mark_price[pair]
            total_debt += self.debt[pair] * self.mark_price[pair]

        margin = 999
        if total_debt != 0:
            margin = total_asset_value / total_debt
        return min(margin, 999)

    def calculate_order_size(self, leverage, symbol):
        if self.is_liquidated():
            return 0

        order_size = 0

        if leverage > 0:
            if self.debt[symbol] > 0:
                order_size += self.debt[symbol]

        elif leverage < 0:
            if self.wallet[symbol] > 0:
                order_size -= self.wallet[symbol]

        equity = 0.0
        for wallet_symbol in self.wallet:
            equity += (self.wallet[wallet_symbol] - self.debt[wallet_symbol]) * self.mark_price[wallet_symbol]
        order_size += leverage * equity / self.mark_price[symbol]

        return order_size

    def order(self, order_size, symbol, fee):
        # print('market_order', wallet, pos_price, pos_contracts, order_contracts, mark_price)
        if self.is_liquidated():
            return 0

        if order_size < 0 and self.wallet[symbol] < -order_size:
            print("err", order_size, symbol, fee)

        self.wallet[symbol] += order_size
        self.wallet_usdt -= order_size * self.mark_price[symbol]
        self.wallet_usdt -= abs(order_size) * self.mark_price[symbol] * fee

        # Repay debt
        """
        if order_size > 0:
            if self.debt[symbol] > 0:
                if self.wallet[symbol] > self.debt[symbol]:
                    self.wallet[symbol] -= self.debt[symbol]
                    self.debt[symbol] = 0
                else:
                    self.debt[symbol] -= self.wallet[symbol]
                    self.wallet[symbol] = 0

        elif order_size < 0:
            if self.debt['usdt'] > 0:
                if self.wallet['usdt'] > self.debt['usdt']:
                    self.wallet['usdt'] -= self.debt['usdt']
                    self.debt['usdt'] = 0
                else:
                    self.debt['usdt'] -= self.wallet['usdt']
                    self.wallet['usdt'] = 0

        # Borrow
        if self.wallet['usdt'] < 0:
            print("BORROW!", self.wallet)
            self.debt['usdt'] = -self.wallet['usdt']
            self.wallet['usdt'] = 0
        if self.wallet[symbol] < 0:
            self.debt[symbol] = -self.wallet[symbol]
            self.wallet[symbol] = 0

        if self.calculate_margin() < self.liquidation_level:
            for wallet_symbol in self.wallet:
                self.wallet[wallet_symbol] = 0.0
                self.debt[wallet_symbol] = 0.0
        """

    def limit_order(self, order_size, symbol):
        self.order(order_size, symbol, -0.00025)

    def market_order(self, order_size, symbol):
        self.order(order_size, symbol, 0.0)  # 0.00075)
        return True

    def sell_pair(self, symbol):
        if self.wallet[symbol] > 0:
            self.market_order(-self.wallet[symbol], symbol)

        elif self.debt[symbol] > 0:
            self.market_order(self.debt[symbol], symbol)


if __name__ == '__main__':

    def atest_order(symbol, order_size, mark_price):
        print("----")
        if order_size > 0:
            print(f'Buy {order_size} {symbol} @ {mark_price}')
        else:
            print(f'Sell {-order_size} {symbol} @ {mark_price}')
        sim.set_mark_price(pair=symbol, mark_price=mark_price)
        sim.market_order(order_size=order_size, symbol=symbol)
        value = sim.get_value_usdt()
        leverage = sim.calculate_leverage()
        margin = sim.calculate_margin()
        print(f'w_usdt: {sim.wallet["usdt"]}, w_{symbol}: {sim.wallet[symbol]}, d_usdt: {sim.debt["usdt"]}, d_{symbol}: {sim.debt[symbol]}')
        print(f'value: {value}, leverage: {leverage}, margin: {margin}')

    def atest_mark_price(symbol, mark_price):
        print("----")
        print(f'Mark price {mark_price}')
        sim.set_mark_price(pair=symbol, mark_price=mark_price)
        value = sim.get_value_usdt()
        leverage = sim.calculate_leverage()
        margin = sim.calculate_margin()
        print(f'w_usdt: {sim.wallet["usdt"]}, w_{symbol}: {sim.wallet[symbol]}, d_usdt: {sim.debt["usdt"]}, d_{symbol}: {sim.debt[symbol]}')
        print(f'value: {value}, leverage: {leverage}, margin: {margin}')

    sim = BinanceSimulator(initial_usdt=10000, symbols=['btcusdt', 'ethusdt'])

    atest_mark_price(symbol='btcusdt', mark_price=10000)
    atest_order(symbol='btcusdt', order_size=1.0, mark_price=10000)
    atest_order(symbol='btcusdt', order_size=2.0, mark_price=10000)
    atest_mark_price(symbol='btcusdt', mark_price=12000)
    atest_mark_price(symbol='btcusdt', mark_price=15000)
    atest_mark_price(symbol='btcusdt', mark_price=8000)
    atest_order(symbol='btcusdt', order_size=-1.0, mark_price=8000)
    atest_order(symbol='btcusdt', order_size=-2.0, mark_price=8000)
    atest_order(symbol='btcusdt', order_size=-2.0, mark_price=10000)
    atest_mark_price(symbol='btcusdt', mark_price=8000)
    atest_mark_price(symbol='btcusdt', mark_price=11000)
