#include "pch.h"
#include "Portfolio.h"


void Portfolio::add_order(sptrOrder order)
{
    orders.push_back(order);
}

void Portfolio::evaluate_orders(void)
{
    for (const auto& order : orders) {
        if (order->state == Order::State::Filled) {
            printf("Filled\n");
        }
    }

    orders.erase(
        std::remove_if(
            orders.begin(),
            orders.end(),
            [](const sptrOrder& order) { return order->state == Order::State::Canceled || order->state == Order::State::Filled; }
        ),
        orders.end()
    );
}

int Portfolio::get_position_count(const Symbol& symbol)
{
    return (int) std::count_if(positions.begin(), positions.end(), [symbol](const auto& position) {return position.symbol == symbol; });
}

int Portfolio::get_order_count(const Symbol& symbol)
{
    return (int) std::count_if(orders.begin(), orders.end(), [symbol](const auto& order) {return order->symbol == symbol; });
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
}

