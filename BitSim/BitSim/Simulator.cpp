#include "pch.h"
#include "Simulator.h"


Simulator::Simulator(void)
{
    wallet_usdt = 10000;
    for (const auto& symbol : symbols) {
        wallet[symbol.idx] = 0;
        mark_price[symbol.idx] = 0;
    }
}

void Simulator::set_mark_price(const Klines& klines)
{
    for (const auto& symbol : symbols) {
        mark_price[symbol.idx] = klines.get_open_price(symbol);
    }
}

double Simulator::get_equity(void) const
{
    double equity = wallet_usdt;
    for (const auto& symbol : symbols) {
        equity += wallet[symbol.idx] * mark_price[symbol.idx];
    }
    return equity;
}

double Simulator::get_cash(void) const
{
    return wallet_usdt;
}

void Simulator::limit_order(double position_size, const Symbol& symbol)
{
    limit_orders.emplace_back(symbol, mark_price[symbol.idx], position_size);
}

uptrPositions Simulator::evaluate_limit_orders(const Klines& klines, time_point_ms timestamp)
{
    auto positions = std::make_unique<std::vector<Position>>();
    for (const auto& limit_order : limit_orders) {
        if (klines.get_low_price(limit_order.symbol) < limit_order.price) {
            printf("%s Execute limit order\n", date::format("%F %T", timestamp).c_str());
        }
    }
    return positions;
}
