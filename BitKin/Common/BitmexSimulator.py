

class BitmexSimulator:
    def __init__(self, settings, mark_price, initial_leverage=0):
        self.settings = settings
        self.wallet = 1.0
        self.price = mark_price
        self.contracts = initial_leverage * self.wallet * self.price

    def get_leverage(self, mark_price):
        if self.wallet == 0:
            return 0
        value = self.contracts / mark_price
        leverage = value / self.wallet
        return leverage

    def get_value(self, mark_price):
        if self.wallet == 0:
            return 0
        upnl = self.contracts * (1 / self.price - 1 / mark_price)
        return (self.wallet + upnl) * mark_price

    def calculate_order_size(self, leverage, mark_price):
        if self.wallet == 0.0:
            return 0.0

        position_margin = 0.0
        position_leverage = 0.0
        upnl = 0.0

        if self.contracts != 0.0:
            sign = self.contracts / abs(self.contracts)
            entry_value = abs(self.contracts / self.price)
            mark_value = abs(self.contracts / mark_price)
            upnl = sign * (entry_value - mark_value)
            position_margin = max(0.0, abs(self.contracts / self.price) - upnl)
            position_leverage = sign * position_margin / self.wallet

        max_contracts = self.settings['max_leverage'] * (self.wallet + upnl) * mark_price
        margin = self.wallet * min(max(leverage, -self.settings['max_leverage']), self.settings['max_leverage'])
        contracts = min(max(margin * mark_price, -max_contracts), max_contracts)

        order_contracts = contracts - self.contracts
        return order_contracts

    def order(self, order_contracts, mark_price, fee):
        # print('market_order', wallet, pos_price, pos_contracts, order_contracts, mark_price)
        if self.wallet == 0:
            return

        upnl = self.contracts * (1 / self.price - 1 / mark_price)

        max_contracts = self.settings['max_leverage'] * (self.wallet + upnl) * mark_price
        if self.contracts + order_contracts > max_contracts or self.contracts + order_contracts < -max_contracts:
            order_contracts = max_contracts - self.contracts

        # Fee
        self.wallet -= fee * abs(order_contracts) / mark_price

        # Realised profit and loss
        # Wallet only changes when abs(contracts) decrease
        #realised = self.wallet
        if (self.contracts > 0) and (order_contracts < 0):
            self.wallet += (1 / self.price - 1 / mark_price) * min(-order_contracts, self.contracts)
        elif (self.contracts < 0) and (order_contracts > 0):
            self.wallet += (1 / self.price - 1 / mark_price) * max(-order_contracts, self.contracts)

        #realised = self.wallet - realised
        #if realised != 0:
        #    print("realise", self.wallet, realised, fee, realised - fee)

        # Calculate average entry price
        if (self.contracts >= 0 and order_contracts > 0) or (self.contracts <= 0 and order_contracts < 0):
            self.price = (self.contracts * self.price + order_contracts * mark_price) / (self.contracts + order_contracts)
        elif (self.contracts >= 0 and self.contracts + order_contracts < 0) or (self.contracts <= 0 and self.contracts + order_contracts > 0):
            self.price = mark_price

        #print(f"t({datetime.fromtimestamp(timestamp)}) price({mark_price}) contr({order_contracts})")

        # Calculate position contracts
        self.contracts += order_contracts


    def limit_order(self, order_contracts, mark_price):
        self.order(order_contracts, mark_price, -0.0025)

    def market_order(self, order_contracts, mark_price):
        self.order(order_contracts, mark_price, 0.0075)
