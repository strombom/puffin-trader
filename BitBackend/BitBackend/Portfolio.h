#pragma once
#include "precompiled_headers.h"

#include "BitLib/Uuid.h"
#include "BitLib/DateTime.h"
#include "Symbols.h"
#include "Common.h"

#include <list>
#include <mutex>


class Portfolio
{
public:
    Portfolio(void) :
        wallet_balance(0.0), wallet_available(0.0) {}

    struct Order
    {
        Order(Uuid id, Symbol symbol, Side side, double qty, double price, time_point_us created, bool confirmed) :
            id(id), symbol(symbol), side(side), qty(qty), price(price), created(created), confirmed(confirmed) {}

        Uuid id;
        Symbol symbol;
        Side side;
        double qty;
        double price;
        time_point_us created;
        bool confirmed;

        Order& operator=(const Order& order) {
            if (this != &order) {
                id = order.id;
                symbol = order.symbol;
                side = order.side;
                qty = order.qty;
                price = order.price;
                created = order.created;
                confirmed = order.confirmed;
            }
            return *this;
        }

    private:
        friend bool operator==(const Order& l, const Order& r) { return l.id == r.id; }
    };

    struct Position
    {
        Position(void) : 
            qty(0.0) {}

        Position(double qty) :
            qty(qty) {}

        double qty;
    };

    void update_order(Uuid id, const Symbol& symbol, Side side, double qty, double price, time_point_us created, bool confirmed);
    void update_position(const Symbol& symbol, Side side, double qty);
    void update_wallet(double balance, double available);
    void new_trade(const Symbol& symbol, Side side, double price);

    void debug_print(void);

private:
    std::mutex orders_mutex;
    std::mutex positions_mutex;

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
