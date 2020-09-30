#pragma once
#include "pch.h"

#include "LiveData.h"
#include "MT_Policy.h"
#include "BitmexAccount.h"
#include "BitmexRestApi.h"
#include "BitmexWebSocket.h"

#include <thread>


enum TraderState 
{ 
    start,
    wait_for_next_agg_tick,
    wait_for_next_agg_tick_worker,
    delete_orders,
    place_new_order,
    place_new_order_work,
    order_monitoring
};

class BitmexTrader
{
public:
    //BitmexTrader(void);
    BitmexTrader(sptrLiveData live_data, sptrMT_Policy mt_policy);

    void start(void);
    void shutdown(void);

private:
    TraderState trader_state;

    time_point_ms current_interval_timestamp;
    torch::Tensor current_interval_feature;

    bool new_order_first_try;
    double action_leverage;
    double action_stop_loss;
    double action_take_profit;
    double order_price;
    double order_ask_price;
    double order_bid_price;

    sptrLiveData live_data;
    sptrMT_Policy mt_policy;
    sptrBitmexWebSocket bitmex_websocket;
    sptrBitmexAccount bitmex_account;
    sptrBitmexRestApi bitmex_rest_api;

    std::atomic_bool trader_thread_running;
    std::unique_ptr<std::thread> trader_thread;

    //bool limit_order(void);
    bool market_order(void);
    void trader_worker(void);
};
