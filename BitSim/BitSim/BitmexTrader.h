#pragma once
#include "pch.h"

#include "BitmexAccount.h"
#include "BitmexRestApi.h"
#include "BitmexWebSocket.h"

#include <thread>


enum TraderState 
{ 
    start,
    wait_for_next_interval,
    bitbot_action,
    delete_orders,
    delete_orders_worker,
    place_new_order,
    place_new_order_work,
    order_monitoring
};

class BitmexTrader
{
public:
    BitmexTrader(void);

    void start(void);
    void shutdown(void);

private:
    TraderState trader_state;
    time_point_ms start_timestamp;
    int delete_orders_remaining_retries;
    bool new_order_first_try;
    double desired_ask_price;
    double desired_bid_price;
    double desired_leverage;
    double order_price;
    double order_ask_price;
    double order_bid_price;

    sptrBitmexWebSocket bitmex_websocket;
    sptrBitmexAccount bitmex_account;
    sptrBitmexRestApi bitmex_rest_api;

    std::atomic_bool trader_thread_running;
    std::unique_ptr<std::thread> trader_thread;

    bool limit_order(void);
    void trader_worker(void);
};
