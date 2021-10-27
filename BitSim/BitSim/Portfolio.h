#pragma once
#include "Order.h"
#include "Klines.h"
#include "Symbols.h"
#include "Position.h"
#include "Simulator.h"
#include "BitLib/BitBotConstants.h"


class Portfolio
{
public:
    //void add_order(sptrOrder order);
    void set_mark_prices(const Klines& klines);
    void evaluate_positions(void);
    bool has_available_position(const Symbol& symbol);
    bool has_available_order(const Symbol& symbol);
    void cancel_oldest_order(const Symbol& symbol);
    void place_limit_order(time_point_ms timestamp, const Symbol& symbol, double position_size);
    void evaluate_orders(time_point_ms timestamp, const Klines& klines);

    double get_equity(void) const;
    double get_cash(void) const;

private:
    Simulator simulator;
    std::vector<Position> positions;
    std::vector<sptrOrder> orders;

    inline int get_position_count(const Symbol& symbol);
    inline int get_order_count(const Symbol& symbol);
};
