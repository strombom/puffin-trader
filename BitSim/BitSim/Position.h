#pragma once
#include "Order.h"
#include "Symbols.h"
#include "BitLib/DateTime.h"


struct Position
{
    enum class State { Opening, Active, Closing};

    Position(time_point_ms created, int delta_idx, sptrOrder order) :
        created(created), state(State::Opening), symbol(order->symbol), delta_idx(delta_idx), created_price(order->price), filled_price(0), take_profit(0), stop_loss(0), amount(order->amount), order(order) {}

    time_point_ms created;
    State state;
    Symbol symbol;

    int delta_idx;
    double created_price;
    double filled_price;
    double take_profit;
    double stop_loss;
    double amount;

    sptrOrder order;
};
