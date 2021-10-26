#pragma once
#include "Symbols.h"
#include "BitLib/DateTime.h"


struct Order
{
public:
    enum class State { Active, Filled, Canceled };
    enum class Side { Buy, Sell };

    Order(time_point_ms created, const Symbol& symbol, Order::Side side, double price, double amount) :
        state(State::Active), created(created), symbol(symbol), side(side), price(price), amount(amount), cancel(false) {}

    time_point_ms created;
    Symbol symbol;
    State state;
    Side side;
    double price;
    double amount;
    bool cancel;
};

using sptrOrder = std::shared_ptr<Order>;
