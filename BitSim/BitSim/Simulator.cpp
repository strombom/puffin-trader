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
    const auto price = mark_price[symbol.idx] - symbol.tick_size;
    position_size = int(position_size / symbol.min_qty) * symbol.min_qty;
    limit_orders.emplace_back(symbol, price, position_size);
}

uptrOrders Simulator::evaluate_orders(const Klines& klines, time_point_ms timestamp)
{
    auto executed_orders = std::make_unique<std::vector<Order>>();
    for (auto& limit_order : limit_orders) {
        if (klines.get_low_price(limit_order.symbol) < limit_order.price) {
            printf(
                "%s Execute limit order %s %.5f %.5f\n", 
                date::format("%F %T", timestamp).c_str(),
                limit_order.symbol.name.data(),
                limit_order.price,
                limit_order.amount
            );
            executed_orders->emplace_back(limit_order.to_order());
            limit_order.executed = true;
        }
    }

    if (executed_orders->size() > 0) {
        // Remove executed limit orders
        limit_orders.erase(
            std::remove_if(
                limit_orders.begin(), 
                limit_orders.end(), 
                [&](const LimitOrder& limit_order) 
                { 
                    return limit_order.executed; 
                }
            ), 
            limit_orders.end()
        );
    }
    
    return executed_orders;
}
