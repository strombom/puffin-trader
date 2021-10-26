#pragma once
#include "Klines.h"
#include "Order.h"
#include "Position.h"
#include "Symbols.h"
#include "BitLib/BitBotConstants.h"
#include "BitLib/Utils.h"

#include <memory>

/*
struct LimitOrder
{
    LimitOrder(const Symbol& symbol, double price, double amount) : 
        order(std::make_shared<Order>(symbol, price, amount))
    {
        //order = std::make_shared<Order>(symbol, price, amount);
    }

    sptrOrder order;
};
*/

class Simulator
{
public:
    Simulator(void);

    void set_mark_price(const Klines& klines);

    double get_equity(void) const;
    double get_cash(void) const;
    sptrOrder limit_order(time_point_ms timestamp, const Symbol& symbol, double position_size);
    //void evaluate_order
    void evaluate_orders(time_point_ms timestamp, const Klines& klines);
    void cancel_orders(void);

private:
    float wallet_usdt;
    std::array<double, symbols.size()> wallet;
    std::array<double, symbols.size()> mark_price;

    std::vector<sptrOrder> limit_orders;
    //UuidGenerator uuid_generator;
};
