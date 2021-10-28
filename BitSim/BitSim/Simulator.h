#pragma once
#include "Klines.h"
#include "Order.h"
#include "Position.h"
#include "Symbols.h"
#include "BitLib/BitBotConstants.h"
#include "BitLib/Utils.h"

#include <memory>


class Simulator
{
public:
    Simulator(void);

    void set_mark_prices(const Klines& klines);
    double get_mark_price(const Symbol& symbol) const;
    double get_wallet_usdt(void) const;
    double get_wallet(const Symbol& symbol) const;

    double get_total_equity(void) const;

    sptrOrder limit_order(time_point_ms timestamp, const Symbol& symbol, double price, double quantity);
    void adjust_order_volumes(void);

    void evaluate_orders(time_point_ms timestamp, const Klines& klines);
    void cancel_orders(void);

private:
    std::vector<sptrOrder> limit_orders;
    //UuidGenerator uuid_generator;

    double wallet_usdt;
    std::array<double, symbols.size()> wallet;
    std::array<double, symbols.size()> mark_price;
};
