#pragma once
#include "precompiled_headers.h"

#include "OrderBook.h"
#include "Portfolio.h"


class OrderManager
{
public:
    OrderManager(sptrPortfolio portfolio, sptrOrderBooks order_books) : 
        portfolio(portfolio), order_books(order_books) {}

private:
    sptrPortfolio portfolio;
    sptrOrderBooks order_books;
};
