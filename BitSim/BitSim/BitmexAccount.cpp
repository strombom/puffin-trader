#include "pch.h"
#include "BitmexAccount.h"

BitmexAccount::BitmexAccount(void)
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

void BitmexAccount::insert_order(const std::string& order_id, const std::string& symbol, time_point_ms timestamp, bool buy, int size, double price)
{

}

void BitmexAccount::fill_order(const std::string& order_id, const std::string& symbol, time_point_ms timestamp, int remaining_size)
{

}

void BitmexAccount::delete_order(const std::string& order_id)
{

}

void BitmexAccount::margin_update(double leverage)
{

}

void BitmexAccount::set_price(double price)
{

}
