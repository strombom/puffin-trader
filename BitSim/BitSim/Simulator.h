#pragma once
#include "Klines.h"
#include "BitLib/BitBotConstants.h"


class Simulator
{
public:
    Simulator(void);

    void set_mark_price(const Klines& klines);

    double get_equity(void) const;
    double get_cash(void) const;

private:
    float wallet_usdt;
    std::array<double, BitBot::symbols.size()> wallet;
    std::array<double, BitBot::symbols.size()> mark_price;
};
