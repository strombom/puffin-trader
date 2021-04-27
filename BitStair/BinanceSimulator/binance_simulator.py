

class BinanceSimulator:
    def __init__(self, initial_usdt, pairs, max_leverage):
        self.max_leverage = max_leverage
        self.liquidation_level = 1.1
        self.pairs = pairs
        self.wallet = {'usdt': initial_usdt}
        self.debt = {'usdt': 0.0}
        self.mark_price = {}
        for pair in self.pairs:
            self.wallet[pair] = 0.0
            self.debt[pair] = 0.0
            self.mark_price[pair] = 0.0

    def is_liquidated(self):
        return False
        # if self.wallet_btc == 0 and self.wallet_usdt == 0:
        #    return True
        # else:
        #    return False

    def calculate_leverage(self):
        if self.is_liquidated():
            return -1

        total_asset_value = self.wallet_usdt
        total_debt = self.debt_usdt
        for pair in self.wallet:
            total_asset_value += self.wallet[pair] * self.mark_price[pair]
            total_debt += self.debt[pair] * self.mark_price[pair]
        total_equity = total_asset_value - total_debt

        if total_debt >= total_asset_value:
            for pair in self.wallet:
                self.wallet[pair] = 0.0
                self.debt[pair] = 0.0
            return -1

        return total_asset_value / total_equity - 1

    def set_mark_price(self, pair, mark_price):
        self.mark_price[pair] = mark_price

    def get_value_usdt(self):
        value = self.wallet_usdt - self.debt_usdt
        for pair in self.pairs:
            value += (self.wallet[pair] - self.debt[pair]) * self.mark_price[pair]
        return value

    def calculate_margin(self):
        total_asset_value = self.wallet['usdt']
        total_debt = self.debt['usdt']
        for pair in self.pairs:
            total_asset_value += self.wallet[pair] * self.mark_price[pair]
            total_debt += self.debt[pair] * self.mark_price[pair]

        margin = 999
        if total_debt != 0:
            margin = total_asset_value / total_debt
        return min(margin, 999)

    def calculate_order_size(self, leverage, pair):
        if self.is_liquidated():
            return 0

        order_size = 0

        if leverage > 0:
            if self.debt[pair] > 0:
                order_size += self.debt[pair]

        elif leverage < 0:
            if self.wallet[pair] > 0:
                order_size -= self.wallet[pair]

        equity = self.wallet_usdt - self.debt_usdt + (self.wallet_btc - self.debt_btc) * mark_price
        order_size += leverage * equity / mark_price

        return order_size

    def order(self, order_size_btc, mark_price, fee):
        # print('market_order', wallet, pos_price, pos_contracts, order_contracts, mark_price)
        if self.is_liquidated(mark_price=mark_price):
            return 0

        fee_usdt = order_size_btc * mark_price * fee

        self.wallet_btc += order_size_btc
        self.wallet_usdt -= order_size_btc * mark_price
        self.wallet_usdt -= fee_usdt

        # Repay debt
        if order_size_btc > 0:
            if self.debt_btc > 0:
                if self.wallet_btc > self.debt_btc:
                    self.wallet_btc -= self.debt_btc
                    self.debt_btc = 0
                else:
                    self.debt_btc -= self.wallet_btc
                    self.wallet_btc = 0

        elif order_size_btc < 0:
            if self.debt_usdt > 0:
                if self.wallet_usdt > self.debt_usdt:
                    self.wallet_usdt -= self.debt_usdt
                    self.debt_usdt = 0
                else:
                    self.debt_usdt -= self.wallet_usdt
                    self.wallet_usdt = 0

        # Borrow
        if self.wallet_usdt < 0:
            self.debt_usdt = -self.wallet_usdt
            self.wallet_usdt = 0
        if self.wallet_btc < 0:
            self.debt_btc = -self.wallet_btc
            self.wallet_btc = 0

        if self.calculate_margin(mark_price=mark_price) < self.liquidation_level:
            self.wallet_usdt = 0.0
            self.wallet_btc = 0.0
            self.debt_usdt = 0.0
            self.debt_btc = 0.0

    def limit_order(self, order_size_btc, mark_price):
        self.order(order_size_btc, mark_price, 0.00075)

    def market_order(self, order_size_btc, mark_price):
        self.order(order_size_btc, mark_price, 0.00075)


if __name__ == '__main__':

    def test_order(order_size_btc, mark_price):
        print("----")
        if order_size_btc > 0:
            print(f'Buy {order_size_btc} btc @ {mark_price}')
        else:
            print(f'Sell {-order_size_btc} btc @ {mark_price}')
        sim.market_order(order_size_btc=order_size_btc, mark_price=mark_price)
        value = sim.get_value_usdt(mark_price=mark_price)
        leverage = sim.calculate_leverage(mark_price=mark_price)
        margin = sim.calculate_margin(mark_price=mark_price)
        print(f'w_usdt: {sim.wallet_usdt}, w_btc: {sim.wallet_btc}, d_usdt: {sim.debt_usdt}, d_btc: {sim.debt_btc}')
        print(f'value: {value}, leverage: {leverage}, margin: {margin}')

    def test_mark_price(mark_price):
        print("----")
        print(f'Mark price {mark_price}')
        value = sim.get_value_usdt(mark_price=mark_price)
        leverage = sim.calculate_leverage(mark_price=mark_price)
        margin = sim.calculate_margin(mark_price=mark_price)
        print(f'w_usdt: {sim.wallet_usdt}, w_btc: {sim.wallet_btc}, d_usdt: {sim.debt_usdt}, d_btc: {sim.debt_btc}')
        print(f'value: {value}, leverage: {leverage}, margin: {margin}')


    mark_price = 10000
    sim = BinanceSimulator(initial_usdt=10000, initial_btc=0, max_leverage=2, mark_price=mark_price, initial_leverage=0.0)

    test_mark_price(mark_price=mark_price)
    test_order(order_size_btc=1.0, mark_price=10000)
    test_order(order_size_btc=2.0, mark_price=10000)
    test_mark_price(mark_price=12000)
    test_mark_price(mark_price=15000)
    test_mark_price(mark_price=8000)
    test_order(order_size_btc=-1.0, mark_price=8000)
    test_order(order_size_btc=-2.0, mark_price=8000)
    test_order(order_size_btc=-2.0, mark_price=10000)
    test_mark_price(mark_price=8000)
    test_mark_price(mark_price=11000)

