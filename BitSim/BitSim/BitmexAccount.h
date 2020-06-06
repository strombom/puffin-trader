#pragma once
#include "pch.h"

#include "DateTime.h"


class BitmexAccount
{
public:
    BitmexAccount(void);

    void insert_order(const std::string& symbol, const std::string& order_id, time_point_ms timestamp, bool buy, int size, double price);
    void fill_order(const std::string& symbol, const std::string& order_id, time_point_ms timestamp, int remaining_size);
    void delete_order(const std::string& order_id);
    void set_leverage(double mark_value);
    void set_wallet(double amount);
    void set_price(const std::string& symbol, double price);

    double get_leverage(void);
    double get_price(void);

    void limit_order(double order_leverage);
    void market_order(double order_leverage);

private:
    double position_leverage;
    double last_price;
    double wallet;
};

using sptrBitmexAccount = std::shared_ptr<BitmexAccount>;
