#pragma once
#include "Order.h"
#include "Symbols.h"
#include "BitLib/DateTime.h"
#include "BitLib/Utils.h"


struct Position
{
    enum class State { Opening, Active, Closing, Closed };

    Position(time_point_ms created, int delta_idx, sptrOrder order);

    time_point_ms created;
    Uuid uuid;
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
