#pragma once
#include "Klines.h"
#include "BitLib/BitBotConstants.h"


class Simulator
{
public:
    Simulator(void);

    void set_mark_price(const Klines& klines);

private:
    float wallet_usdt;
    std::array<float, BitBot::symbols.size()> wallet;
    std::array<float, BitBot::symbols.size()> mark_price;
};
