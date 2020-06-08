#include "pch.h"

#include "BitmexAccount.h"
#include "Logger.h"


BitmexOrder::BitmexOrder(time_point_ms timestamp, int size) :
    timestamp(timestamp), buy(false), size(size), price(0.0), valid(false)
{

}

BitmexOrder::BitmexOrder(time_point_ms timestamp, bool buy, int size, double price) :
    timestamp(timestamp), buy(buy), size(size), price(price), valid(true)
{

}

void BitmexOrder::set_size(time_point_ms fill_timestamp, int remaining_size)
{
    timestamp = fill_timestamp;
    size = remaining_size;
}

void BitmexOrder::set_price(time_point_ms fill_timestamp, double _price)
{
    timestamp = fill_timestamp;
    price = _price;
}

BitmexAccount::BitmexAccount(void) :
    leverage(0.0), contracts(0), upnl(0.0), last_price(0.0), wallet(0.0), active_order("")
{

}

int BitmexAccount::count_orders(void)
{
    return (int)orders.size();
}

void BitmexAccount::clear_orders(void)
{
    orders.clear();
}

void BitmexAccount::print_orders(void)
{
    for (auto const& [key, val] : orders) {
        logger.info("BitmexAccount::print_orders [%s %s b(%d) s(%d) p(%0.1f) v(%d)]", key.c_str(), DateTime::to_string(val->timestamp).c_str(), val->buy, val->size, val->price, val->valid);
    }
}

void BitmexAccount::insert_order(const std::string& symbol, const order_id_t& order_id, time_point_ms timestamp, bool buy, int size, double price)
{
    auto slock = std::scoped_lock{ order_mutex };

    if (symbol != "XBTUSD") {
        return;
    }

    logger.info("BitmexAccount::insert_order %s %s %0.1f %d", order_id.c_str(), buy ? "Buy" : "Sell", price, size);

    orders.insert_or_assign(order_id, std::make_unique<BitmexOrder>(timestamp, buy, size, price));
    print_orders();
}

void BitmexAccount::fill_order(const std::string& symbol, const order_id_t& order_id, time_point_ms timestamp, int remaining_size)
{
    auto slock = std::scoped_lock{ order_mutex };

    if (symbol != "XBTUSD") {
        return;
    }

    logger.info("BitmexAccount::fill_order %s %d", order_id.c_str(), remaining_size);

    if (orders.count(order_id) == 0 && remaining_size > 0) {
        // Order does not exist, insert new order with unknown direction and price
        orders.insert_or_assign(order_id, std::make_unique<BitmexOrder>(timestamp, remaining_size));
        print_orders();
        return;
    }

    if (remaining_size == 0) {
        orders.erase(order_id);
    }
    else {
        orders.at(order_id)->set_size(timestamp, remaining_size);
    }
    print_orders();
}

void BitmexAccount::delete_order(const order_id_t& order_id)
{
    auto slock = std::scoped_lock{ order_mutex };

    logger.info("BitmexAccount::delete_order %s", order_id.c_str());

    if (orders.count(order_id) > 0) {
        orders.erase(order_id);
    }
    print_orders();
}

void BitmexAccount::amend_order(const std::string& symbol, const order_id_t& order_id, time_point_ms timestamp, int size, double price)
{
    if (symbol != "XBTUSD") {
        return;
    }

    if (orders.count(order_id) > 0) {
        orders.at(order_id)->set_size(timestamp, size);
        orders.at(order_id)->set_price(timestamp, price);
    }
    print_orders();
}

void BitmexAccount::amend_order_size(const std::string& symbol, const order_id_t& order_id, time_point_ms timestamp, int size)
{
    if (symbol != "XBTUSD") {
        return;
    }

    if (orders.count(order_id) > 0) {
        orders.at(order_id)->set_size(timestamp, size);
    }
    print_orders();
}

void BitmexAccount::amend_order_price(const std::string& symbol, const order_id_t& order_id, time_point_ms timestamp, double price)
{
    if (symbol != "XBTUSD") {
        return;
    }

    if (orders.count(order_id) > 0) {
        orders.at(order_id)->set_price(timestamp, price);
    }
    print_orders();
}

void BitmexAccount::set_contracts(int _contracts)
{
    contracts = _contracts;
}

void BitmexAccount::set_leverage(double mark_value)
{
    if (wallet == 0) {
        leverage = 0;
    }
    else {
        leverage = -1e-8 * mark_value / wallet;
    }
    //logger.info("BitmexAccount::set_leverage %0.4f", position_leverage);
}

void BitmexAccount::set_upnl(double _upnl)
{
    upnl = _upnl;
}

void BitmexAccount::set_wallet(double _wallet)
{
    wallet = _wallet;
    //logger.info("BitmexAccount::set_wallet %0.4f", wallet);
}

void BitmexAccount::set_price(double price)
{
    if (last_price != price) {
        //logger.info("BitmexAccount::set_price set_price %0.1f", price);
    }
    last_price = price;
}

void BitmexAccount::set_ask_price(double price)
{
    ask_price = price;
}

void BitmexAccount::set_bid_price(double price)
{
    bid_price = price;
}

int BitmexAccount::get_contracts(void) const
{
    return contracts;
}

double BitmexAccount::get_leverage(void) const
{
    return leverage;
}

double BitmexAccount::get_upnl(void) const
{
    return upnl;
}

double BitmexAccount::get_wallet(void) const
{
    return wallet;
}

double BitmexAccount::get_price(void) const
{
    return last_price;
}

double BitmexAccount::get_ask_price(void) const
{
    return ask_price;
}

double BitmexAccount::get_bid_price(void) const
{
    return bid_price;
}
