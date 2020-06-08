#include "pch.h"

#include "Logger.h"
#include "BitmexTrader.h"
#include "BitBotConstants.h"


BitmexTrader::BitmexTrader(void) :
    trader_thread_running(true),
    trader_state(TraderState::wait_for_next_interval),
    order_leverage(0.0),
    delete_orders_remaining_retries(0),
    new_order_first_try(true),
    order_mark_price(0)
{
    bitmex_account = std::make_shared<BitmexAccount>();
    bitmex_rest_api = std::make_shared<BitmexRestApi>(bitmex_account);
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
        continue;
        /*
        static auto first = true;
        if (first) {
            auto success = bitmex_rest_api->limit_order(1, 9200);
            std::this_thread::sleep_for(1000ms);
            success = bitmex_rest_api->limit_order(-1, 9900);
            std::this_thread::sleep_for(1000ms);
            success = bitmex_rest_api->delete_all();
            std::this_thread::sleep_for(1000ms);
            success = bitmex_rest_api->limit_order(1, 9400);
            first = false;
        }
        order_leverage = 0.02;
        order_mark_price = bitmex_account->get_price();
        std::this_thread::sleep_for(100ms);
        continue;
        */

        if (trader_state == TraderState::start) {
            bitmex_rest_api->delete_all();
            trader_state = TraderState::wait_for_next_interval;
        }
        else if (trader_state == TraderState::wait_for_next_interval) {
            static auto first = true;
            if (first) {
                first = false;
                std::this_thread::sleep_for(2000ms);
                trader_state = TraderState::bitbot_action;
            }
            else {
                std::this_thread::sleep_for(500ms);
            }
        }
        else if (trader_state == TraderState::bitbot_action) {
            start_timestamp = system_clock_ms_now();
            const auto place_order = true;
            if (place_order) {
                order_leverage = 0.02;
                order_mark_price = bitmex_account->get_price();
                trader_state = TraderState::delete_orders;
            }
            else {
                trader_state = TraderState::wait_for_next_interval;
            }
        }
        else if (trader_state == TraderState::delete_orders) {
            delete_orders_remaining_retries = 3;
            trader_state = TraderState::delete_orders_worker;
        }
        else if (trader_state == TraderState::delete_orders_worker) {
            const auto success = bitmex_rest_api->delete_all();
            if (success) {
                trader_state = TraderState::place_new_order;
            }
            else {
                delete_orders_remaining_retries--;
                if (delete_orders_remaining_retries == 0 || system_clock_ms_now() - start_timestamp > 2s) {
                    // No retries left or timeout
                    trader_state = TraderState::wait_for_next_interval;
                }
                else {
                    // Try again after delay
                    std::this_thread::sleep_for(500ms);
                }
            }
        }
        else if (trader_state == TraderState::place_new_order) {
            new_order_first_try = true;
            trader_state = TraderState::place_new_order_work;
        }
        else if (trader_state == TraderState::place_new_order_work) {
            const auto success = limit_order();
            if (success) {
                std::this_thread::sleep_for(1s);
                trader_state = TraderState::order_monitoring;
            }
            else {
                if (system_clock_ms_now() - start_timestamp > 5s) {
                    trader_state = TraderState::wait_for_next_interval;
                }
                else if (new_order_first_try) {
                    new_order_first_try = false;
                }
                else {
                    std::this_thread::sleep_for(1s);
                }
            }
        }
        else if (trader_state == TraderState::order_monitoring) {
            std::this_thread::sleep_for(150ms);
            if (system_clock_ms_now() - start_timestamp > 5s) {
                trader_state = TraderState::wait_for_next_interval;
            }
            else if (bitmex_account->count_orders() == 0) {
                trader_state = TraderState::wait_for_next_interval;
            } 
            else if (bitmex_account->get_price() != order_mark_price) {
                trader_state = TraderState::delete_orders;
            }
        }
    }
}

bool BitmexTrader::limit_order(void)
{
    const auto position_contracts = bitmex_account->get_contracts();
    const auto mark_price = bitmex_account->get_price();
    const auto wallet = bitmex_account->get_wallet();
    const auto upnl = bitmex_account->get_upnl();

    if (wallet == 0.0) {
        return false;
    }

    const auto max_contracts = BitSim::BitMex::max_leverage * (wallet + upnl) * mark_price;
    const auto margin = wallet * std::clamp(order_leverage, -BitSim::BitMex::max_leverage, BitSim::BitMex::max_leverage);
    const auto contracts = std::clamp(margin * mark_price, -max_contracts, max_contracts);
    const auto order_contracts = int(contracts - position_contracts);

    auto success = false;
    if (order_contracts > 0) {
        success = bitmex_rest_api->limit_order(order_contracts, mark_price - 0.5);
    }
    else if (order_contracts < 0) {
        success = bitmex_rest_api->limit_order(order_contracts, mark_price + 0.5);
    }

    //logger.info("order leverage(%f) pos_contracts(%d) contracts(%d) price(%0.1f)", order_leverage, position_contracts, order_contracts, mark_price);

    return success;
}
