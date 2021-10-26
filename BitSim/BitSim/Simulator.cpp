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

sptrOrder Simulator::limit_order(time_point_ms timestamp, const Symbol& symbol, double position_size)
{
    const auto price = mark_price[symbol.idx] - symbol.tick_size;
    const auto quantity = int(position_size / symbol.min_qty) * symbol.min_qty;

    auto order = std::make_shared<Order>(timestamp, symbol, Order::Side::Buy, price, quantity);

    limit_orders.emplace_back(order);

    //const auto order_id = uuid_generator.generate();

    //printf("%s\n", order_id.to_string().c_str());

    return order;
}

void Simulator::cancel_orders(void)
{
    for (auto& limit_order : limit_orders) {
        if (limit_order->cancel) {
            limit_order->state = Order::State::Canceled;
        }
    }

    limit_orders.erase(
        std::remove_if(
            limit_orders.begin(),
            limit_orders.end(),
            [](const sptrOrder& limit_order) { return limit_order->state == Order::State::Canceled; }
        ),
        limit_orders.end()
    );
}

void Simulator::evaluate_orders(time_point_ms timestamp, const Klines& klines)
{
    for (auto& limit_order : limit_orders) {
        if (limit_order->side == Order::Side::Buy && klines.get_low_price(limit_order->symbol) < limit_order->price) {
            limit_order->state = Order::State::Filled;
        }
    }

    //auto executed_orders = std::make_unique<std::vector<uptrOrder>>();
    /*
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
    */
}
