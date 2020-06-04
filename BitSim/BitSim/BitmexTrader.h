#pragma once
#include "pch.h"

#include <thread>

#include "BitmexWebSocket.h"


class BitmexTrader
{
public:
    BitmexTrader(void);

    void start(void);
    void shutdown(void);

private:
    sptrBitmexWebSocket websocket;

    std::atomic_bool trader_thread_running;
    std::unique_ptr<std::thread> trader_thread;

    void trader_worker(void);
};
