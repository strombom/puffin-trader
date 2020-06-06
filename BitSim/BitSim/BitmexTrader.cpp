#include "pch.h"

#include "Logger.h"
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
        std::this_thread::sleep_for(1ms); // 3500ms);

        static auto first = true;
        if (first) {
            first = false;

            bitmex_rest_api.limit_order(0.02);
            /*
            limit_order(0.0);
            limit_order(0.03);
            limit_order(-0.03);
            limit_order(1.0);
            limit_order(-1.0);
            */
        }


    }
}

void BitmexTrader::limit_order(double order_leverage)
{
    const auto position_contracts = bitmex_account->get_contracts();
    const auto mark_price = bitmex_account->get_price();
    const auto wallet = bitmex_account->get_wallet();
    const auto upnl = bitmex_account->get_upnl();

    if (wallet == 0.0) {
        return;
    }

    const auto max_contracts = BitSim::BitMex::max_leverage * (wallet + upnl) * mark_price;
    const auto margin = wallet * std::clamp(order_leverage, -BitSim::BitMex::max_leverage, BitSim::BitMex::max_leverage);
    const auto contracts = std::clamp(margin * mark_price, -max_contracts, max_contracts);
    const auto order_contracts = int(contracts - position_contracts);

    bitmex_rest_api.limit_order(order_contracts);
    //logger.info("order leverage(%f) pos_contracts(%d) contracts(%d) price(%0.1f)", order_leverage, position_contracts, order_contracts, mark_price);
}

void BitmexTrader::market_order(double order_leverage)
{

}
