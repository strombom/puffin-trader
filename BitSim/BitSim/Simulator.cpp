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

double Simulator::get_equity(void) const
{
    double equity = 0.0;
    for (const auto& symbol : BitBot::symbols) {
        equity += wallet[symbol.idx] * mark_price[symbol.idx];
    }
    return equity;
}

double Simulator::get_cash(void) const
{
    return wallet_usdt;
}
