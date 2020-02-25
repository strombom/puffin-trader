
"""
Wallet Balance 0,4144 XBT
Unrealised PNL -0,0035 XBT
Margin Balance 0,4104 XBT
----------------------------
Position Margin 0,0111 XBT        
Order Margin 0,0000 XBT           # Initial margin
Available Balance 0,3993 XBT
----------------------------
3% Margin Used 2.5x Leverage

Position
--------
Position size = 10000
Value = 1.0372
Entry price = 9674.92
Mark price = 9641.85
Liq price = 6941
Margin = 0.0111
UPnL(RoE) = -0.0033 (-32.88%)
RPnL = -0.0011
"""

#Entry value = 10000/9674.92=1.03360
#Position Margin = Entry value / leverage + -0.0033 = 0.0103714 = 0.0070714


# Wallet Balance = Deposits - Withdrawals + Realised PNL
# Margin Balance = Wallet Balance + Unrealised PNL
# Available Balance = Margin Balance - Order Margin - Position Margin


taker_fee = 0.075 / 100
maintenance_rate = 0.5 / 100


class Simulator:
    def __init__(self, wallet=1.0):
        self.wallet = wallet
        self.entry_price = 0.0
        self.n_contracts = 0.0 # +long -short
        self.position_rpnl = 0.0
        self.margin = 100


    #def _calc(self, price):
    #    position_value = self.position_size / self.entry_price
    #    bankruptcy_price = (1 - 1 / (1 + (self.position_size / (self.wallet * self.entry_price)))) * self.entry_price
    #    print("Bankrupcy price:", bankruptcy_price)
    #    maintenance_margin = abs(0.005 * self.position_size / price)
    #    taker_fee = abs(0.00075 * self.position_size / bankruptcy_price)
    #    liquidation_price = 1 / (-((self.wallet - maintenance_margin - taker_fee) / (-self.position_size) - 1 / self.entry_price))
    #    print("Liqudation price:", liquidation_price)

    def buy(self, n_contracts, price):
        self._order(n_contracts, price)

    def sell(self, n_contracts, price):
        self._order(-n_contracts, price)

    def _order(self, n_contracts, price):
        value = n_contracts / price
        if n_contracts > 0:
            print("Buy: {:.1f} contracts at {:.2f} USD/BTC => {:.4f} BTC".format(n_contracts, price, value))
        else:
            print("Sell: {:.1f} contracts at {:.2f} USD/BTC => {:.4f} BTC".format(n_contracts, price, value))

        if self.n_contracts == 0:
            upnl = 0
        else:
            upnl = (1/self.entry_price - 1/price) * self.n_contracts

        if (self.n_contracts >= 0 and n_contracts > 0) or (self.n_contracts <= 0 and n_contracts < 0):
            self.entry_price = (self.n_contracts * self.entry_price + n_contracts * price) / (self.n_contracts + n_contracts)

        elif (self.n_contracts >= 0 and self.n_contracts + n_contracts < 0) or (self.n_contracts <= 0 and self.n_contracts + n_contracts > 0):
            self.entry_price = price

        self.n_contracts += n_contracts

        self.wallet -= abs(n_contracts / price) * taker_fee + upnl
        self.position_rpnl += abs(n_contracts / price) * taker_fee

    def print_balance(self, price):
        print("  Wallet balance: {}".format(self.wallet))
        if self.n_contracts != 0:
            upnl = (1/self.entry_price - 1/price) * self.n_contracts
            entry_value = abs(self.n_contracts / self.entry_price)
            maintenance_margin = maintenance_rate * entry_value
            liquidation_fee = taker_fee * (self.wallet + entry_value)
            liquidation_price = self.n_contracts * self.entry_price / (self.entry_price * (self.wallet - maintenance_margin - liquidation_fee) + self.n_contracts)
            if liquidation_price < 0:
                if self.n_contracts < 0:
                    liquidation_price = 100000000
                else:
                    liquidation_price = 0
            print("  Entry price: {}".format(self.entry_price))
            print("  Contracts: {}".format(self.n_contracts))
            print("  Liquidation price: {}".format(liquidation_price))
            print("  UPnL: {}".format(upnl))
            print("  RPnL: {}".format(self.position_rpnl*1000))
        print("====")


    #def calc_liquidation_price(avg_price, contracts, balance):
    #    position_value = contracts / avg_price
    #    bankruptcy_price = (1 - 1 / (1 + (contracts / (balance * avg_price)))) * avg_price
    #    maintenance_margin = abs(0.005 * position_value)
    #    taker_fee = abs(0.00075 * contracts / bankruptcy_price)
    #    liquidation_price = 1 / (-((balance - maintenance_margin - taker_fee) / (-contracts) - 1 / avg_price))
    #    return liquidation_price

sim = Simulator(wallet=0.420481)

sim.sell(n_contracts=10000.0, price=9622.79)
sim.print_balance(price=9646.02)
sim.buy(n_contracts=5000.0, price=9678)
sim.print_balance(price=9648.32)
sim.buy(n_contracts=10000.0, price=9681.48)
sim.print_balance(price=9649.0)
sim.buy(n_contracts=1000.0, price=9659.0)
sim.print_balance(price=9652.67)
sim.sell(n_contracts=7000.0, price=9663.0)
sim.print_balance(price=9673.08)
#sim.print_balance(7000.0)




