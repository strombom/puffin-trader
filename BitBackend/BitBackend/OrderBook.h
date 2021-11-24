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

    bool updated(void);

    double get_last_ask(void);
    double get_last_bid(void);

    enum class ChangeSource
    {
        trade,
        order_book
    };

    struct Entry
    {
        double price;
        double qty;
    };

    static const int size = 25;
    std::array<Entry*, size> asks;
    std::array<Entry*, size> bids;

private:
    double latest_ask;
    double latest_bid;
};

using sptrOrderBooks = std::shared_ptr<std::array<OrderBook, symbols.size()>>;

sptrOrderBooks makeOrderBooks(void);
