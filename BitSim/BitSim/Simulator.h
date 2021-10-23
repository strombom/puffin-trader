#pragma once
#include "Klines.h"
#include "Position.h"
#include "Symbols.h"
#include "BitLib/BitBotConstants.h"


struct LimitOrder
{
    LimitOrder(const Symbol& symbol, double price, double amount) :
        symbol(symbol), price(price), amount(amount), executed(false) {}

    //LimitOrder(const LimitOrder& limit_order) :
    //    symbol(limit_order.symbol), price(limit_order.price), amount(limit_order.amount), executed(limit_order.executed) { }

    //LimitOrder(const LimitOrder& other);

/*
LimitOrder::LimitOrder(const LimitOrder& other)
    : symbol(other.symbol)
{
    //std::swap(price, other.price);
    //symbol = other.symbol;
    price = other.price;
    amount = other.amount;
    executed = other.executed;
}
    LimitOrder& operator=(LimitOrder&& a) noexcept
    {
        // Self-assignment detection
        if (&a == this)
            return *this;

        symbol = a.symbol;
        // Release any resource we're holding
        //delete m_ptr;

        // Transfer ownership of a.m_ptr to m_ptr
        //m_ptr = a.m_ptr;
        //a.m_ptr = nullptr; // we'll talk more about this line below

        return *this;
    }
*/

    //LimitOrder& operator=(LimitOrder&&) = default;

    Symbol symbol;
    double price;
    double amount;
    bool executed;
};

using uptrLimitOrders = std::unique_ptr<std::vector<LimitOrder>>;

class Simulator
{
public:
    Simulator(void);

    void set_mark_price(const Klines& klines);

    double get_equity(void) const;
    double get_cash(void) const;
    void limit_order(double position_size, const Symbol& symbol);
    uptrLimitOrders evaluate_limit_orders(const Klines& klines, time_point_ms timestamp);

private:
    float wallet_usdt;
    std::array<double, symbols.size()> wallet;
    std::array<double, symbols.size()> mark_price;

    std::vector<LimitOrder> limit_orders;
};
