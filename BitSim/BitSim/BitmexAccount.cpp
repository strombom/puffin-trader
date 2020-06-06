#include "pch.h"

#include "BitmexAccount.h"
#include "Logger.h"


BitmexAccount::BitmexAccount(void) :
    position_leverage(0.0), last_price(0.0), wallet(0.0)
{

}

double BitmexAccount::get_leverage(void)
{
    return position_leverage;
}

double BitmexAccount::get_price(void)
{
    return last_price;
}

void BitmexAccount::limit_order(double leverage)
{

}

void BitmexAccount::market_order(double leverage)
{

}

void BitmexAccount::insert_order(const std::string& symbol, const std::string& order_id, time_point_ms timestamp, bool buy, int size, double price)
{
    if (symbol != "XBTUSD") {
        return;
    }

    logger.info("BitmexAccount::insert_order %s %s %0.1f %d", order_id.c_str(), buy ? "Buy" : "Sell", price, size);
}

void BitmexAccount::fill_order(const std::string& symbol, const std::string& order_id, time_point_ms timestamp, int remaining_size)
{
    if (symbol != "XBTUSD") {
        return;
    }

    logger.info("BitmexAccount::fill_order %s %d", order_id.c_str(), remaining_size);
}

void BitmexAccount::delete_order(const std::string& order_id)
{
    logger.info("BitmexAccount::delete_order %s", order_id.c_str());
}

void BitmexAccount::set_leverage(double mark_value)
{
    if (wallet == 0) {
        position_leverage = 0;
    }
    else {
        position_leverage = -1e-8 * mark_value / wallet;
    }
    logger.info("BitmexAccount::set_leverage %0.4f", position_leverage);
}

void BitmexAccount::set_wallet(double amount)
{
    wallet = amount;
    logger.info("BitmexAccount::set_wallet %0.4f", wallet);
}

void BitmexAccount::set_price(const std::string& symbol, double price)
{
    if (symbol != "XBTUSD") {
        return;
    }
    if (last_price != price) {
        //logger.info("BitmexAccount::set_price set_price %0.1f", price);
    }
    last_price = price;
}
