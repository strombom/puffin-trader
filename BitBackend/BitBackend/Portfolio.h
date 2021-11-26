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
    Portfolio(void);

    enum class OrderState
    {
        unconfirmed,
        confirmed,
        replacing
    };

    struct Order
    {
        Order(Uuid id, Symbol symbol, Side side, double qty, double price, time_point_us created, OrderState state) :
            id(id), symbol(symbol), side(side), qty(qty), price(price), created(created), state(state) {}

        Uuid id;
        Symbol symbol;
        Side side;
        double qty;
        double price;
        time_point_us created;
        //bool confirmed;
        //double replacing_qty;
        //double replacing_price;
        OrderState state;

        Order& operator=(const Order& order) {
            if (this != &order) {
                id = order.id;
                symbol = order.symbol;
                side = order.side;
                qty = order.qty;
                price = order.price;
                created = order.created;
                state = order.state;

                //confirmed = order.confirmed;
                //replacing_qty = order.replacing_qty;
                //replacing_price = order.replacing_price;
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

    void create_order(Uuid id, const Symbol& symbol, Side side, double qty, double price, time_point_us created);
    void update_order(Uuid id, const Symbol& symbol, Side side, double qty, double price, time_point_us updated);
    void replace_order(Uuid id, double qty, double price);
    void remove_order(Uuid id);

    void update_position(const Symbol& symbol, Side side, double qty);
    void update_wallet(double balance, double available);
    void new_trade(const Symbol& symbol, Side side, double price);

    void debug_print(void);

    std::array<std::list<Order>, symbols.size()> orders;
    std::array<Position, symbols.size()> positions_buy;
    std::array<Position, symbols.size()> positions_sell;

private:
    std::mutex orders_mutex;
    std::mutex positions_mutex;

    double wallet_balance;
    double wallet_available;

    std::array<double, symbols.size()> last_bid;

    std::tuple<const Symbol*, Order*> find_order(Uuid id);
    Order* find_order(const Symbol& symbol, Uuid id);
};

using sptrPortfolio = std::shared_ptr<Portfolio>;
