#pragma once
#include "Order.h"
#include "Symbols.h"
#include "Position.h"
#include "BitLib/BitBotConstants.h"


class Portfolio
{
public:
    void add_order(sptrOrder order);

    void evaluate_orders(void);

    bool has_available_position(const Symbol& symbol);
    bool has_available_order(const Symbol& symbol);
    void cancel_oldest_order(const Symbol& symbol);

private:
    std::vector<Position> positions;
    std::vector<sptrOrder> orders;

    int get_position_count(const Symbol& symbol);
    int get_order_count(const Symbol& symbol);
};
