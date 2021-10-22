#include "pch.h"
#include "Simulator.h"


Simulator::Simulator(void)
{
    wallet_usdt = 10000;
    for (const auto& symbol : BitBot::symbols) {
        wallet[symbol.idx] = 0;
        mark_price[symbol.idx] = 0;
    }
}

void Simulator::set_mark_price(const Klines& klines)
{
    for (const auto& symbol : BitBot::symbols) {
        mark_price[symbol.idx] = klines.get_open_price(symbol);
    }
}
