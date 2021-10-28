#include "pch.h"
#include "Order.h"


Order::Order(time_point_ms created, const Symbol& symbol, Order::Side side, double price, double amount) :
    state(State::Active), uuid(uuid_generator.generate()), created(created), symbol(symbol), side(side), price(price), amount(amount), cancel(false)
{
}
