#pragma once
#include "Order.h"
#include "Symbols.h"
#include "BitLib/DateTime.h"


struct Position
{
    Position(time_point_ms timestamp, const Symbol& symbol, double price, double amount) : 
        timestamp(timestamp), symbol(symbol) {}

    const time_point_ms timestamp;

    const Symbol& symbol;
    double amount;

    double mark_price;
    double take_profit;
    double stop_loss;
};
