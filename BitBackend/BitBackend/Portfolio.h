#pragma once
#include "precompiled_headers.h"

#include "BitLib/Uuid.h"
#include "BitLib/DateTime.h"
#include "Symbols.h"


enum class Side { buy, sell };


class OrderBook
{
public:
    OrderBook(void);

    void clear(void);
    void insert(double price, Side side, double qty);
    void update(double price, Side side, double qty);
    void del(double price, Side side);

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
};

class Portfolio
{
public:
    Portfolio(void) :
        wallet_balance(0.0), wallet_available(0.0) {}

    struct Order
    {
        Order(Uuid uuid, Symbol symbol, double qty, double price, Side side, time_point_us created) :
            uuid(uuid), symbol(symbol), qty(qty), price(price), side(side), created(created) {}

        Uuid uuid;
        Symbol symbol;
        double qty;
        double price;
        Side side;
        time_point_us created;
    };

    struct Position
    {
        Position(void) : 
            qty(0.0) {}

        Position(double qty) :
            qty(qty) {}

        double qty;
    };

    void update_order(Uuid id, const Symbol& symbol, Side side, double price, double qty, std::string status, time_point_us created);
    void update_position(const Symbol& symbol, Side side, double qty);
    void update_wallet(double balance, double available);
    void new_trade(const Symbol& symbol, Side side, double price);
    void order_book_clear(const Symbol& symbol);
    void order_book_insert(const Symbol& symbol, double price, Side side, double qty);
    void order_book_update(const Symbol& symbol, double price, Side side, double qty);
    void order_book_delete(const Symbol& symbol, double price, Side side);
    double order_book_get_last_bid(const Symbol& symbol);

private:
    std::map<Uuid, Order> orders;
    std::array<Position, symbols.size()> positions_buy;
    std::array<Position, symbols.size()> positions_sell;
    double wallet_balance;
    double wallet_available;
    std::array<double, symbols.size()> last_trade_price;
    std::array<double, symbols.size()> last_ask;
    std::array<double, symbols.size()> last_bid;
    std::array<OrderBook, symbols.size()> order_books;
};

using sptrPortfolio = std::shared_ptr<Portfolio>;
