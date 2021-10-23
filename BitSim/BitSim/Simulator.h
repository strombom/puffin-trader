#pragma once
#include "Klines.h"
#include "Position.h"
#include "Symbols.h"
#include "BitLib/BitBotConstants.h"


struct LimitOrder
{
    LimitOrder(const Symbol& symbol, double price, double amount) :
        symbol(symbol), price(price), amount(amount) {}

    const Symbol& symbol;
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
    void limit_order(double position_size, const Symbol& symbol);
    uptrPositions evaluate_limit_orders(const Klines& klines, time_point_ms timestamp);

private:
    float wallet_usdt;
    std::array<double, symbols.size()> wallet;
    std::array<double, symbols.size()> mark_price;

    std::vector<LimitOrder> limit_orders;
};
