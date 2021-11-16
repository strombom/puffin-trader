#pragma once
#include "precompiled_headers.h"

#include "BitLib/Uuid.h"
#include "BitLib/DateTime.h"
#include "Symbols.h"
#include "Common.h"


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

private:
    std::map<Uuid, Order> orders;
    std::array<Position, symbols.size()> positions_buy;
    std::array<Position, symbols.size()> positions_sell;
    double wallet_balance;
    double wallet_available;
    std::array<double, symbols.size()> last_trade_price;
    std::array<double, symbols.size()> last_ask;
    std::array<double, symbols.size()> last_bid;
};

using sptrPortfolio = std::shared_ptr<Portfolio>;
