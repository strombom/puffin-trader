#pragma once
#include "Symbols.h"
#include "BitLib/DateTime.h"
#include "BitLib/Utils.h"


struct Order
{
public:
    enum class State { Active, Filled, Canceled };
    enum class Side { Buy, Sell };

    Order(time_point_ms created, const Symbol& symbol, Order::Side side, double price, double amount);

    time_point_ms created;
    Uuid uuid;
    Symbol symbol;
    State state;
    Side side;
    double price;
    double amount;
    bool cancel;
};

using sptrOrder = std::shared_ptr<Order>;
