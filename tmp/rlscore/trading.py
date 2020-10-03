

class Position:
    def __init__(self, mark_price, initial_leverage, take_profit, min_profit):
        self.wallet = 1.0
        self.price = mark_price
        self.contracts = 0 #initial_leverage * self.wallet * self.price
        self.take_profit = take_profit
        self.min_profit = min_profit
        self._update()

    def _update(self):
        if self.contracts == 0:
            self.direction = 1
        else:
            self.direction = self.contracts / abs(self.contracts)
        self.take_profit_price = self.price * (1 + self.direction * take_profit)
        self.stop_loss_price = self.price * (1 - self.direction * stop_loss)

        if self.wallet < 0.2:
            self.wallet = 0
            self.contracts = 0
            return

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

    def market_order(self, order_leverage, mark_price, timestamp):

        # print('market_order', wallet, pos_price, pos_contracts, order_contracts, mark_price)
        if self.wallet == 0:
            return

        upnl = self.contracts * (1 / self.price - 1 / mark_price)

        #if timestamp > 1578035300:
        #    print("a")
        #elif timestamp > 1578416400:
        #    print("a")

        max_contracts = settings['max_leverage'] * (self.wallet + upnl) * mark_price
        max_contracts = min(max_contracts, settings['max_order_value'] * mark_price)

        margin = self.wallet * min(max(order_leverage, -settings['max_leverage']), settings['max_leverage'])
        order_contracts = min(max(margin * mark_price, -max_contracts), max_contracts) - self.contracts

        # Fee
        fee = taker_fee * abs(order_contracts) / mark_price
        self.wallet -= fee

        # Realised profit and loss
        # Wallet only changes when abs(contracts) decrease
        realised = self.wallet
        if (self.contracts > 0) and (order_contracts < 0):
            self.wallet += (1 / self.price - 1 / mark_price) * min(-order_contracts, self.contracts)
        elif (self.contracts < 0) and (order_contracts > 0):
            self.wallet += (1 / self.price - 1 / mark_price) * max(-order_contracts, self.contracts)

        realised = self.wallet - realised
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
        self._update()
