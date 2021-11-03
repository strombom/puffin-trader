#pragma once
#include "precompiled_headers.h"

#include "BitLib/Uuid.h"
#include "BitLib/DateTime.h"
#include "Symbols.h"


class Portfolio
{
public:
    Portfolio(void);

    enum class Side { buy, sell };

    struct Order
    {
        Order(Uuid uuid, Symbol symbol, double qty, double price, Side side) :
            uuid(uuid), symbol(symbol), qty(qty), price(price), side(side) {}

        Uuid uuid;
        Symbol symbol;
        double qty;
        double price;
        Side side;
    };

    struct Position
    {
        Position(Uuid uuid, Symbol symbol, double size, Side side) :
            uuid(uuid), symbol(symbol), size(size), side(side) {}

        Uuid uuid;
        Symbol symbol;
        double size;
        Side side;
    };

    void update_order(Uuid id, const Symbol& symbol, Portfolio::Side side, double price, double qty, std::string status, time_point_us timestamp);

private:
    std::map<Uuid, Order> orders;
    std::map<Uuid, Position> positions;

};

using sptrPortfolio = std::shared_ptr<Portfolio>;
