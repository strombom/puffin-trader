#include "pch.h"

#include "BitmexTrader.h"
#include "BitBotConstants.h"


BitmexTrader::BitmexTrader(void) :
    trader_thread_running(true)
{
    bitmex_account = std::make_shared<BitmexAccount>();
    bitmex_websocket = std::make_shared<BitmexWebSocket>(bitmex_account);
}

void BitmexTrader::start(void)
{
    bitmex_websocket->start();
    trader_thread = std::make_unique<std::thread>(&BitmexTrader::trader_worker, this);
}

void BitmexTrader::shutdown(void)
{
    std::cout << "BitmexTrader: Shutting down" << std::endl;
    bitmex_websocket->shutdown();
    trader_thread_running = false;

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
