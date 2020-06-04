#pragma once
#include "pch.h"

#include <thread>


class BitmexTrader
{
public:
    BitmexTrader(void);

    void start(void);
    void shutdown(void);

private:

    std::atomic_bool trader_thread_running;
    std::unique_ptr<std::thread> trader_thread;

    void trader_worker(void);
};
