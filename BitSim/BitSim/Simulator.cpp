#include "pch.h"
#include "Simulator.h"


Simulator::Simulator(void)
{
    wallet_usdt = 10000;
    for (const auto& symbol : BitSim::symbols) {
        wallet[symbol.idx] = 0;
        mark_price[symbol.idx] = 0;
    }
}

void Simulator::set_mark_price(const Klines& klines)
{
    for (const auto& symbol : BitSim::symbols) {
        mark_price[symbol.idx] = klines.get_open_price(symbol);
    }
}

double Simulator::get_equity(void) const
{
    double equity = 0.0;
    for (const auto& symbol : BitSim::symbols) {
        equity += wallet[symbol.idx] * mark_price[symbol.idx];
    }
    return equity;
}

double Simulator::get_cash(void) const
{
    return wallet_usdt;
}

void Simulator::limit_order(double position_size, const BitSim::Symbol& symbol)
{
    limit_orders.emplace_back(symbol, position_size, mark_price[symbol.idx]);
}
