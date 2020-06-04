#include "pch.h"

#include "BitmexTrader.h"
#include "BitBotConstants.h"


BitmexTrader::BitmexTrader(void) :
    trader_thread_running(true)
{
    bitmex_account = std::make_unique<BitmexAccount>();
}

void BitmexTrader::start(void)
{
    // Start websocket worker
    trader_thread = std::make_unique<std::thread>(&BitmexTrader::trader_worker, this);
    bitmex_account->start();
}

void BitmexTrader::shutdown(void)
{
    std::cout << "BitmexTrader: Shutting down" << std::endl;
    trader_thread_running = false;

    bitmex_account->shutdown();

    try {
        trader_thread->join();
    }
    catch (...) {}
}

void BitmexTrader::trader_worker(void)
{
    while (trader_thread_running) {
        std::this_thread::sleep_for(500ms);
    }
}
