#pragma once
#include "pch.h"

#include <thread>

#include "BitmexAccount.h"
#include "BitmexRestApi.h"
#include "BitmexWebSocket.h"


class BitmexTrader
{
public:
    BitmexTrader(void);

    void start(void);
    void shutdown(void);

private:
    sptrBitmexWebSocket bitmex_websocket;
    sptrBitmexAccount bitmex_account;

    std::atomic_bool trader_thread_running;
    std::unique_ptr<std::thread> trader_thread;

    void limit_order(double order_leverage);
    void market_order(double order_leverage);

    void trader_worker(void);
};
