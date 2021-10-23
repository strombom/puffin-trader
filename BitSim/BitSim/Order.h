#pragma once
#include "Symbols.h"


struct Order
{
    Order(const Symbol& symbol, double price, double amount) :
        symbol(symbol), price(price), amount(amount) {}

    Symbol symbol;
    double price;
    double amount;
};

using uptrOrders = std::unique_ptr<std::vector<Order>>;
