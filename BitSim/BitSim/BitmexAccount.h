#pragma once
#include "pch.h"

#include "DateTime.h"


class BitmexAccountOrder
{
public:
    BitmexAccountOrder(time_point_ms timestamp, int size);
    BitmexAccountOrder(time_point_ms timestamp, bool buy, int size, double price);

    void fill(time_point_ms fill_timestamp, int remaining_size);

    time_point_ms timestamp;
    bool buy;
    int size;
    double price;
    bool valid;

private:

};

using uptrBitmexAccountOrder = std::unique_ptr<BitmexAccountOrder>;

class BitmexAccount
{
private:
    using order_id_t = std::string;

public:
    BitmexAccount(void);

    void insert_order(const std::string& symbol, const order_id_t& order_id, time_point_ms timestamp, bool buy, int size, double price);
    void fill_order(const std::string& symbol, const order_id_t& order_id, time_point_ms timestamp, int remaining_size);
    void delete_order(const order_id_t& order_id);
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
    const order_id_t active_order;

    void print_orders(void);

    std::map<const order_id_t, uptrBitmexAccountOrder> orders;
};

using sptrBitmexAccount = std::shared_ptr<BitmexAccount>;
