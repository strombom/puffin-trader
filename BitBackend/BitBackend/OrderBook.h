#pragma once

#include "Common.h"
#include "Symbols.h"

#include <array>


class OrderBook
{
public:
    OrderBook(void);

    void clear(void);
    void insert(double price, Side side, double qty);
    void update(double price, Side side, double qty);
    void del(double price, Side side);

    double get_last_ask(void);
    double get_last_bid(void);

private:
    struct Entry
    {
        double price;
        double qty;
    };

    static const int size = 25;
    std::array<Entry*, size> asks;
    std::array<Entry*, size> bids;

    double old_last_ask;
    double old_last_bid;
};

using sptrOrderBooks = std::shared_ptr<std::array<OrderBook, symbols.size()>>;

sptrOrderBooks makeOrderBooks(void);
