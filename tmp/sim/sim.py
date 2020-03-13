
taker_fee = 0.075 / 100
maintenance_rate = 0.5 / 100

max_leverage = 10.0
order_hysteresis = 0.1

class Simulator:
    def __init__(self, wallet=1.0):
        self.wallet = wallet
        self.entry_price = 0.0
        self.n_contracts = 0.0 # +long -short
        self.margin = 100

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

        rpnl = -taker_fee * abs(n_contracts / price)

        if self.n_contracts > 0 and n_contracts < 0:
            rpnl += (1/self.entry_price - 1/price) * min(-n_contracts, self.n_contracts)

        elif self.n_contracts < 0 and n_contracts > 0:
            rpnl += (1/self.entry_price - 1/price) * max(-n_contracts, self.n_contracts)

        self.wallet += rpnl
        
        if (self.n_contracts >= 0 and n_contracts > 0) or (self.n_contracts <= 0 and n_contracts < 0):
            self.entry_price = (self.n_contracts * self.entry_price + n_contracts * price) / (self.n_contracts + n_contracts)

        elif (self.n_contracts >= 0 and self.n_contracts + n_contracts < 0) or (self.n_contracts <= 0 and self.n_contracts + n_contracts > 0):
            self.entry_price = price

        self.n_contracts += n_contracts

    def print_balance(self, mark_price):
        print("  Wallet balance: {}".format(self.wallet))
        if self.n_contracts != 0:
            upnl = (1/self.entry_price - 1/mark_price) * self.n_contracts
            entry_value = self.n_contracts / self.entry_price
            maintenance_margin = maintenance_rate * abs(entry_value)
            liquidation_fee = taker_fee * abs(self.wallet + entry_value)
            liquidation_price = self.n_contracts * self.entry_price / (self.entry_price * (self.wallet - maintenance_margin - liquidation_fee) + self.n_contracts)
            if liquidation_price < 0:
                if self.n_contracts < 0:
                    liquidation_price = 100000000
                else:
                    liquidation_price = 0

            print("  Entry price: {}".format(self.entry_price))
            print("  Mark price: {}".format(mark_price))
            print("  Contracts: {}".format(self.n_contracts))
            print("  Liquidation price: {}".format(liquidation_price))
            print("  UPnL: {}".format(upnl*1000))
        print("====")



def marg(wallet, contracts, entry_price, price, buy_size, sell_size):
    print("Wallet:", wallet, "Contracts:", contracts, "Entry price:", entry_price, "Mark price:", price)

    position_margin = 0.0
    position_leverage = 0.0
    upnl = 0.0

    if contracts != 0.0:
        sign = contracts / abs(contracts)
        entry_value = abs(contracts / entry_price)
        mark_value = abs(contracts / price)
        upnl = sign * (entry_value - mark_value)

        if entry_price > 0.0:
            position_margin = max(0.0, abs(contracts / entry_price) - upnl)
        position_leverage = position_margin / wallet

    max_margin = max_leverage * wallet
    available_margin = max_margin - position_margin

    max_contracts = max_leverage * (wallet + upnl) * price

    if contracts > 0.0:
        max_buy_contracts = max(0.0, available_margin * price)
        max_sell_contracts = max(0.0, max_contracts + contracts)
    elif contracts < 0.0:
        max_buy_contracts = max(0.0, max_contracts - contracts)
        max_sell_contracts = max(0.0, available_margin * price)
    else:
        max_buy_contracts = max_margin * price
        max_sell_contracts = max_margin * price

    buy_size = max(0, (buy_size - order_hysteresis) / (1.0 - order_hysteresis))
    sell_size = max(0, (sell_size - order_hysteresis) / (1.0 - order_hysteresis))

    buy_contracts = min(max_buy_contracts, max_contracts * buy_size);
    sell_contracts = min(max_sell_contracts, max_contracts * sell_size);

    print("UPnL", upnl)
    print("Max margin", max_margin)
    print("Position margin", position_margin)
    print("Position leverage", position_leverage)

    print("Available margin", available_margin)
    #print("Max buy contracts", max_buy_contracts)
    print("Max sell contracts", max_sell_contracts)
    #print("Max contracts", max_contracts)
    #print("Buy contracts", buy_contracts)
    #print("Sell contracts", sell_contracts)
    print("---")


marg(wallet=2.0, contracts=1000.0, entry_price=100.0, price=90.0, buy_size=0.5, sell_size=0.0)
marg(wallet=2.0, contracts=1000.0, entry_price=100.0, price=110.0, buy_size=0.5, sell_size=0.0)



sim = Simulator(wallet=2.0)

sim.buy(n_contracts=1000.0, price=100.0)
sim.print_balance(mark_price=110.0)
sim.sell(n_contracts=4200.0, price=110.0)
sim.print_balance(mark_price=110.0)


#marg(wallet=2.0, contracts=0.0, entry_price=100.0, price=110.0, buy_size=1.0, sell_size=0.0)
#marg(wallet=2.0, contracts=0.0, entry_price=100.0, price=100.0, buy_size=1.0, sell_size=0.0)
#marg(wallet=2.0, contracts=0.0, entry_price=100.0, price=90.0, buy_size=1.0, sell_size=0.0)

#marg(wallet=2.0, contracts=2000.0, entry_price=100.0, price=110.0, buy_size=1.0, sell_size=0.0)
#marg(wallet=2.0, contracts=2000.0, entry_price=100.0, price=100.0, buy_size=1.0, sell_size=0.0)
#marg(wallet=2.0, contracts=2000.0, entry_price=100.0, price=90.0, buy_size=1.0, sell_size=0.0)

#marg(wallet=2.0, contracts=-2000.0, entry_price=100.0, price=110.0, buy_size=0.0, sell_size=0.0)
#marg(wallet=2.0, contracts=-2000.0, entry_price=100.0, price=110.0, buy_size=0.05, sell_size=0.0)
#marg(wallet=2.0, contracts=-2000.0, entry_price=100.0, price=110.0, buy_size=0.1, sell_size=0.0)
#marg(wallet=2.0, contracts=-2000.0, entry_price=100.0, price=110.0, buy_size=0.15, sell_size=0.0)
#marg(wallet=2.0, contracts=-2000.0, entry_price=100.0, price=110.0, buy_size=0.20, sell_size=0.0)
#marg(wallet=2.0, contracts=-2000.0, entry_price=100.0, price=110.0, buy_size=1.0, sell_size=0.0)
#marg(wallet=2.0, contracts=-2000.0, entry_price=100.0, price=100.0, buy_size=1.0, sell_size=0.0)
#marg(wallet=2.0, contracts=-2000.0, entry_price=100.0, price=90.0, buy_size=0.05, sell_size=0.0)
#marg(wallet=2.0, contracts=-2000.0, entry_price=100.0, price=90.0, buy_size=0.10, sell_size=0.0)
#marg(wallet=2.0, contracts=-2000.0, entry_price=100.0, price=90.0, buy_size=0.15, sell_size=0.0)
#marg(wallet=2.0, contracts=-2000.0, entry_price=100.0, price=90.0, buy_size=0.9, sell_size=0.0)
#marg(wallet=2.0, contracts=-2000.0, entry_price=100.0, price=90.0, buy_size=1.0, sell_size=0.0)




"""
    const auto price = intervals->rows[intervals_idx].last_price;
    const auto upnl = (1.0 / pos_price + 1 / price) * pos_contracts;
    auto position_margin = 0.0;
    if (pos_contracts != 0) {
        position_margin = std::abs(pos_contracts / pos_price);
    }
    
    const auto available_balance = wallet + upnl - position_margin;

    const auto available_margin = 
"""


quit()






sim = Simulator(wallet=0.420481)

sim.sell(n_contracts=10000.0, price=9622.79)
sim.print_balance(mark_price=9646.02)
sim.buy(n_contracts=5000.0, price=9678)
sim.print_balance(mark_price=9648.32)
sim.buy(n_contracts=10000.0, price=9681.48)
sim.print_balance(mark_price=9649.0)
sim.buy(n_contracts=1000.0, price=9659.0)
sim.print_balance(mark_price=9652.67)
sim.sell(n_contracts=7000.0, price=9663.0)
sim.print_balance(mark_price=9673.08)
sim.buy(n_contracts=1000.0, price=9673.0)
sim.print_balance(mark_price=9673.08)

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

