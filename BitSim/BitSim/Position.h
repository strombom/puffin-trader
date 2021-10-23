#pragma once
#include "Symbols.h"
#include "BitLib/DateTime.h"


class Position
{
public:
    Position(time_point_ms timestamp, const Symbol& symbol, double price, double amount) : 
        timestamp(timestamp), symbol(symbol) {}

private:
    const time_point_ms timestamp;
    const Symbol& symbol;
    double mark_price;
    double take_profit;
    double stop_loss;
    double amount;
};

