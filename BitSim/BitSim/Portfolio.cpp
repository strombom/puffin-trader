#include "pch.h"
#include "Portfolio.h"


void Portfolio::set_mark_prices(const Klines& klines)
{
    simulator.set_mark_prices(klines);
}

void Portfolio::evaluate_positions(void)
{
    for (const auto& position : positions) {
        const auto mark_price = simulator.mark_price[position.symbol.idx];
        if (mark_price < position.stop_loss || mark_price > position.take_profit) {
            printf("sell\n");
        }
    }
}

bool Portfolio::has_available_position(const Symbol& symbol)
{
    return positions.size() < BitSim::Portfolio::total_capacity && get_position_count(symbol) < BitSim::Portfolio::symbol_capacity;
}

bool Portfolio::has_available_order(const Symbol& symbol)
{
    return positions.size() + orders.size() < BitSim::Portfolio::total_capacity && get_position_count(symbol) + get_order_count(symbol) < BitSim::Portfolio::symbol_capacity;
}

void Portfolio::cancel_oldest_order(const Symbol& symbol)
{
    const auto symbol_position_count = get_position_count(symbol);
    const auto symbol_order_count = get_order_count(symbol);

    auto oldest_order = sptrOrder{ nullptr };
    for (const auto& order : orders) {
        if ((order->symbol == symbol || symbol_position_count + symbol_order_count < BitSim::Portfolio::symbol_capacity) && (oldest_order == nullptr || order->created > oldest_order->created)) {
            oldest_order = order;
        }
    }
    oldest_order->cancel = true;

    simulator.cancel_orders();

    orders.erase(
        std::remove_if(
            orders.begin(),
            orders.end(),
            [](const sptrOrder& order) {
                return order->state == Order::State::Canceled;
            }
        ),
        orders.end()
    );
}

void Portfolio::place_limit_order(time_point_ms timestamp, const Symbol& symbol, double position_size)
{
    const auto price = simulator.mark_price[symbol.idx] - symbol.tick_size;
    const auto quantity = int(position_size / symbol.min_qty) * symbol.min_qty;

    auto order = simulator.limit_order(timestamp, symbol, price, quantity);
    orders.emplace_back(order);
}

void Portfolio::evaluate_orders(time_point_ms timestamp, const Klines& klines)
{
    for (const auto& order : orders) {
        if (order->state == Order::State::Active) {
            //const auto order_price =
            printf("Active\n");
        }
    }

    for (const auto& order : orders) {
        if (order->state == Order::State::Filled) {
            printf("Filled\n");
        }
    }

    orders.erase(
        std::remove_if(
            orders.begin(),
            orders.end(),
            [](const sptrOrder& order) {
                return order->state == Order::State::Canceled || order->state == Order::State::Filled;
            }
        ),
        orders.end()
    );
}

double Portfolio::get_equity(void) const
{
    double equity = simulator.wallet_usdt;
    for (const auto& symbol : symbols) {
        equity += simulator.wallet[symbol.idx] * simulator.mark_price[symbol.idx];
    }
    return equity;
}

double Portfolio::get_cash(void) const
{
    return simulator.wallet_usdt;
}

inline int Portfolio::get_position_count(const Symbol& symbol)
{
    return (int)std::count_if(positions.begin(), positions.end(), [symbol](const auto& position) {return position.symbol == symbol; });
}

inline int Portfolio::get_order_count(const Symbol& symbol)
{
    return (int)std::count_if(orders.begin(), orders.end(), [symbol](const auto& order) {return order->symbol == symbol; });
}
