#pragma once
#include "Klines.h"
#include "BitLib/BitBotConstants.h"


class LimitOrder
{
public:
    LimitOrder(const BitSim::Symbol& symbol, double price, double amount) :
        symbol(symbol), price(price), amount(amount) {}

private:
    const BitSim::Symbol& symbol;
    double price;
    double amount;
};

class Simulator
{
public:
    Simulator(void);

    void set_mark_price(const Klines& klines);

    double get_equity(void) const;
    double get_cash(void) const;
    void limit_order(double position_size, const BitSim::Symbol& symbol);

private:
    float wallet_usdt;
    std::array<double, BitSim::symbols.size()> wallet;
    std::array<double, BitSim::symbols.size()> mark_price;

    std::vector<LimitOrder> limit_orders;
};
