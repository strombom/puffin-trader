#include "pch.h"

#include "BitmexTrader.h"
#include "BitBotConstants.h"


BitmexTrader::BitmexTrader(void) :
    trader_thread_running(true)
{
    websocket = std::make_shared<BitmexWebSocket>();
}

void BitmexTrader::start(void)
{
    // Start websocket worker
    trader_thread = std::make_unique<std::thread>(&BitmexTrader::trader_worker, this);
    websocket->start();
}

void BitmexTrader::shutdown(void)
{
    std::cout << "BitmexTrader: Shutting down" << std::endl;
    trader_thread_running = false;

    websocket->shutdown();

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
