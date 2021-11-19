#pragma once
#include "precompiled_headers.h"

#include "BitLib/Uuid.h"
#include "BitLib/DateTime.h"
#include "Symbols.h"
#include "Common.h"

#include <list>


class Portfolio
{
public:
    Portfolio(void) :
        wallet_balance(0.0), wallet_available(0.0) {}

    struct Order
    {
        Order(Uuid id, Symbol symbol, double qty, double price, Side side, time_point_us created, bool confirmed) :
            id(id), symbol(symbol), qty(qty), price(price), side(side), created(created), confirmed(confirmed) {}

        Uuid id;
        Symbol symbol;
        double qty;
        double price;
        Side side;
        time_point_us created;
        bool confirmed;
    };

    struct Position
    {
        Position(void) : 
            qty(0.0) {}

        Position(double qty) :
            qty(qty) {}

        double qty;
    };

    void update_order(Uuid id, const Symbol& symbol, Side side, double price, double qty, time_point_us created, bool confirmed);
    void update_position(const Symbol& symbol, Side side, double qty);
    void update_wallet(double balance, double available);
    void new_trade(const Symbol& symbol, Side side, double price);

    void debug_print(void);

private:
    std::array<std::list<Order>, symbols.size()> orders;
    std::array<Position, symbols.size()> positions_buy;
    std::array<Position, symbols.size()> positions_sell;
    double wallet_balance;
    double wallet_available;
    std::array<double, symbols.size()> last_ask;
    std::array<double, symbols.size()> last_bid;

    Order* find_order(const Symbol& symbol, Uuid id);
};

using sptrPortfolio = std::shared_ptr<Portfolio>;
