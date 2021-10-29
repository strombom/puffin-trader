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

double Simulator::get_wallet_usdt(void) const
{
    return wallet_usdt;
}

double Simulator::get_wallet(const Symbol& symbol) const
{
    return wallet[symbol.idx];
}

double Simulator::get_total_equity(void) const
{
    auto equity = wallet_usdt;
    for (const auto& symbol : symbols) {
        equity += wallet[symbol.idx] * mark_price[symbol.idx];
    }
    return equity;
}

sptrOrder Simulator::limit_order(time_point_ms timestamp, const Symbol& symbol, double price, double quantity)
{
    auto side = Order::Side::Buy;
    if (quantity < 0) {
        side = Order::Side::Sell;
    }

    if (side == Order::Side::Sell && wallet[symbol.idx] + 0.0000001 < -quantity) {
        // Insufficient balance
        return nullptr;
    }

    auto order = std::make_shared<Order>(timestamp, symbol, side, price, quantity);
    limit_orders.emplace_back(order);
    return order;
}

void Simulator::adjust_order_volumes(void)
{
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
    auto active_buy_order_count = 0;
    auto total_buy_order_value = 0;
    for (const auto& limit_order : limit_orders) {
        if (limit_order->state == Order::State::Active && limit_order->side == Order::Side::Buy) {
            total_buy_order_value += limit_order->amount * mark_price[limit_order->symbol.idx];
            active_buy_order_count++;
        }
    }
    if (total_buy_order_value > wallet_usdt * 0.95) {
        const auto order_value = 0.95 * wallet_usdt / active_buy_order_count; //  total_buy_order_value / active_buy_order_count;
        for (auto& limit_order : limit_orders) {
            if (limit_order->state == Order::State::Active && limit_order->side == Order::Side::Buy) {
                limit_order->amount = order_value / mark_price[limit_order->symbol.idx];
            }
        }
    }

    for (auto& limit_order : limit_orders) {
        if (limit_order->side == Order::Side::Buy && klines.get_low_price(limit_order->symbol) < limit_order->price) {
            limit_order->state = Order::State::Filled;
        }
        else if (limit_order->side == Order::Side::Sell && klines.get_high_price(limit_order->symbol) > limit_order->price) {
            limit_order->state = Order::State::Filled;
        }

        if (limit_order->state == Order::State::Filled) {
            wallet[limit_order->symbol.idx] += limit_order->amount;
            wallet_usdt -= limit_order->amount * limit_order->price;
            wallet_usdt -= abs(limit_order->amount) * limit_order->price * BitSim::fee;

            if (wallet_usdt < 0) {
                auto a = 0;
            }

            if (wallet[limit_order->symbol.idx] < -0.001) {
                auto a = 0;
            }
            if (wallet[limit_order->symbol.idx] < 0.000001) {
                wallet[limit_order->symbol.idx] = 0.0;
            }
            wallet[limit_order->symbol.idx] = max(0.0, wallet[limit_order->symbol.idx]);

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
