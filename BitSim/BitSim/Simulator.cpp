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

void Simulator::set_mark_prices(const Klines& klines)
{
    for (const auto& symbol : symbols) {
        mark_price[symbol.idx] = klines.get_open_price(symbol);
    }
}

double Simulator::get_mark_price(const Symbol& symbol) const
{
    return mark_price[symbol.idx];
}

sptrOrder Simulator::limit_order(time_point_ms timestamp, const Symbol& symbol, double price, double quantity)
{
    //const auto order_id = uuid_generator.generate();

    auto order = std::make_shared<Order>(timestamp, symbol, Order::Side::Buy, price, quantity);
    limit_orders.emplace_back(order);
    return order;
}

void Simulator::cancel_orders(void)
{
    for (auto& limit_order : limit_orders) {
        if (limit_order->cancel) {
            limit_order->state = Order::State::Canceled;
        }
    }
}

void Simulator::evaluate_orders(time_point_ms timestamp, const Klines& klines)
{
    for (auto& limit_order : limit_orders) {
        if (limit_order->side == Order::Side::Buy && klines.get_low_price(limit_order->symbol) < limit_order->price) {
            limit_order->state = Order::State::Filled;
        }
        else if (limit_order->side == Order::Side::Sell && klines.get_low_price(limit_order->symbol) > limit_order->price) {
            limit_order->state = Order::State::Filled;
        }
    }

    limit_orders.erase(
        std::remove_if(
            limit_orders.begin(),
            limit_orders.end(),
            [](const sptrOrder& limit_order) { 
                return limit_order->state == Order::State::Canceled || limit_order->state == Order::State::Filled;
            }
        ),
        limit_orders.end()
    );
}
