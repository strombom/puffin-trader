#pragma once
#include "pch.h"

#include "DateTime.h"


class BitmexOrder
{
public:
    BitmexOrder(time_point_ms timestamp, int size);
    BitmexOrder(time_point_ms timestamp, bool buy, int size, double price);

    void set_size(time_point_ms fill_timestamp, int remaining_size);
    void set_price(time_point_ms fill_timestamp, double price);

    time_point_ms timestamp;
    bool buy;
    int size;
    double price;
    bool valid;

private:

};

using uptrBitmexOrder = std::unique_ptr<BitmexOrder>;

class BitmexAccount
{
private:
    using order_id_t = std::string;

public:
    BitmexAccount(void);

    void insert_order(const std::string& symbol, const order_id_t& order_id, time_point_ms timestamp, bool buy, int size, double price);
    void fill_order(const std::string& symbol, const order_id_t& order_id, time_point_ms timestamp, int remaining_size);
    void delete_order(const order_id_t& order_id);
    void amend_order(const std::string& symbol, const order_id_t& order_id, time_point_ms timestamp, int size, double price);
    void amend_order_size(const std::string& symbol, const order_id_t& order_id, time_point_ms timestamp, int size);
    void amend_order_price(const std::string& symbol, const order_id_t& order_id, time_point_ms timestamp, double price);
    int count_orders(void);
    void clear_orders(void);
    
    void set_contracts(int contracts);
    void set_leverage(double mark_value);
    void set_upnl(double upnl);
    void set_wallet(double amount);
    void set_price(double price);

    int get_contracts(void) const;
    double get_leverage(void) const;
    double get_upnl(void) const;
    double get_wallet(void) const;
    double get_price(void) const;

private:
    std::mutex order_mutex;

    int contracts;
    double leverage;
    double upnl;
    double wallet;
    double last_price;

    const order_id_t active_order;
    std::map<const order_id_t, uptrBitmexOrder> orders;

    void print_orders(void);
};

using sptrBitmexAccount = std::shared_ptr<BitmexAccount>;
