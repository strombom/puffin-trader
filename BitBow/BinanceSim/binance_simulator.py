

class BinanceSimulator:
    def __init__(self, max_leverage, mark_price, initial_leverage=0):
        self.max_leverage = max_leverage
        self.liquidation_level = 1.1
        self.wallet_usdt = 1000.0
        self.wallet_btc = 0.0
        self.debt_usdt = 0.0
        self.debt_btc = 0.0
        order_size_btc = self.calculate_order_size_btc(leverage=initial_leverage, mark_price=mark_price)
        self.order(order_size_btc=order_size_btc, mark_price=mark_price, fee=0.0)

    def is_liquidated(self, mark_price):
        if self.wallet_btc == 0 and self.wallet_usdt == 0:
            return True
        else:
            return False

    def get_leverage(self, mark_price):
        if self.is_liquidated(mark_price=mark_price):
            return 0

        value = self.contracts_btc / mark_price
        leverage = value / self.wallet_usdt
        return leverage

    def get_value_usdt(self, mark_price):
        value = self.wallet_usdt - self.debt_usdt
        value += self.wallet_btc * mark_price - self.debt_btc * mark_price
        return value

    def calculate_margin(self, mark_price):
        total_asset_value = self.wallet_usdt + self.wallet_btc * mark_price
        total_debt = self.debt_usdt + self.debt_btc * mark_price

        margin = 999
        if total_debt != 0:
            margin = total_asset_value / total_debt
        return min(margin, 999)

    def calculate_order_size_btc(self, leverage, mark_price):
        if self.is_liquidated(mark_price=mark_price):
            return 0

        order_size = 0

        if leverage > 0:
            if self.debt_btc > 0:
                order_size += self.debt_btc

        elif leverage < 0:
            if self.wallet_btc > 0:
                order_size -= self.wallet_btc

        order_size += leverage * self.wallet_usdt / mark_price

        return order_size

    def order(self, order_size_btc, mark_price, fee):
        # print('market_order', wallet, pos_price, pos_contracts, order_contracts, mark_price)
        if self.is_liquidated(mark_price=mark_price):
            return 0

        fee_usdt = order_size_btc * fee

        if order_size_btc > 0:
            if self.debt_btc > 0:
                # Repay debt
                if order_size_btc > self.debt_btc:
                    order_size_btc -= self.debt_btc
                    self.debt_btc = 0
                else:
                    self.debt_btc -= order_size_btc
                    order_size_btc = 0

        elif order_size_btc < 0:
            if self.debt_usdt > 0:
                # Repay debt
                if -order_size_btc * mark_price > self.debt_usdt:
                    order_size_btc += self.debt_usdt / mark_price
                    self.debt_usdt = 0
                else:
                    self.debt_usdt -= -order_size_btc * mark_price
                    order_size_btc = 0

        self.wallet_btc += order_size_btc
        self.wallet_usdt -= order_size_btc * mark_price
        self.wallet_usdt -= fee_usdt

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
    sim = BinanceSimulator(max_leverage=2, mark_price=10000, initial_leverage=2.0)
    print('value usdt', sim.get_value_usdt(mark_price=10000))

    sim.market_order(order_size_btc=1.0, mark_price=10000)
    value = sim.get_value_usdt(mark_price=10000)
    leverage = sim.get_leverage(mark_price=10000)
    print(f'value: {value}, leverage: {leverage}')

