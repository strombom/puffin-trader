#pragma once
#include "pch.h"

#include "DateTime.h"


class BitmexAccount
{
public:
    BitmexAccount(void);

    void insert_order(const std::string &order_id, const std::string &symbol, bool buy, int size, double price, time_point_ms timestamp);
    void fill_order(const std::string& order_id, int size, int remaining_size);
    void delete_order(const std::string& order_id);

    void margin_update(double leverage);

    void set_price(double price);

    double get_leverage(void);
    double get_price(void);

    void limit_order(double order_leverage);
    void market_order(double order_leverage);

private:
    double position_leverage;
    double last_price;
};

using sptrBitmexAccount = std::shared_ptr<BitmexAccount>;
