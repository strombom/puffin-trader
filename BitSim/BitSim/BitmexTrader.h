#pragma once
#include "pch.h"

#include "LiveData.h"
#include "RL_Policy.h"
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
    //BitmexTrader(void);
    BitmexTrader(sptrLiveData live_data, sptrRL_Policy rl_policy);

    void start(void);
    void shutdown(void);

private:
    TraderState trader_state;

    time_point_ms current_interval_timestamp;
    torch::Tensor current_interval_feature;

    bool new_order_first_try;
    double desired_ask_price;
    double desired_bid_price;
    double desired_leverage;
    double order_price;
    double order_ask_price;
    double order_bid_price;

    sptrLiveData live_data;
    sptrRL_Policy rl_policy;
    sptrBitmexWebSocket bitmex_websocket;
    sptrBitmexAccount bitmex_account;
    sptrBitmexRestApi bitmex_rest_api;

    std::atomic_bool trader_thread_running;
    std::unique_ptr<std::thread> trader_thread;

    bool limit_order(void);
    void trader_worker(void);
};
