#include "pch.h"

#include "BitmexTrader.h"
#include "BitLib/Logger.h"
#include "BitLib/BitBotConstants.h"


BitmexTrader::BitmexTrader(sptrLiveData live_data, sptrMT_Policy mt_policy) :
    trader_thread_running(true),
    trader_state(TraderState::start),
    action_leverage(0.0),
    action_stop_loss(0.0),
    action_take_profit(0.0),
    new_order_first_try(true),
    live_data(live_data),
    mt_policy(mt_policy),
    current_interval_timestamp(system_clock_ms_now())
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
            if (bitmex_account->get_mark_price() != 0.0 &&
                bitmex_account->get_ask_price() != 0.0 &&
                bitmex_account->get_bid_price() != 0.0 &&
                bitmex_account->get_wallet() != 0.0) {
                std::this_thread::sleep_for(100ms);
            }
            else {
                bitmex_rest_api->delete_all();
                std::this_thread::sleep_for(500ms);
                trader_state = TraderState::wait_for_next_agg_tick;
            }
        }
        else if (trader_state == TraderState::wait_for_next_agg_tick) {
            trader_state = TraderState::wait_for_next_agg_tick_worker;
        }
        else if (trader_state == TraderState::wait_for_next_agg_tick_worker) {
            const auto agg_tick = live_data->get_next_agg_tick();
            if (agg_tick != nullptr) {
                const auto [leverage, stop_loss, take_profit] = mt_policy->get_action(agg_tick, bitmex_account->get_leverage());
                action_leverage = leverage;
                action_stop_loss = stop_loss;
                action_take_profit = take_profit;

                if (std::abs(action_leverage) > 0.01) {
                    if (bitmex_account->get_contracts() > 0 && bitmex_account->get_mark_price() < action_stop_loss) {
                        //position.market_order(-action_leverage, position.stop_loss_price);
                        trader_state = TraderState::delete_orders;
                    }
                    else if (bitmex_account->get_contracts() <= 0 && bitmex_account->get_mark_price() > action_stop_loss) {
                        //position.market_order(action_leverage, position.stop_loss_price);
                        trader_state = TraderState::delete_orders;
                    }
                    else if (bitmex_account->get_contracts() > 0 && bitmex_account->get_mark_price() > action_take_profit) {
                        //position.market_order(-action_leverage, event.price);
                        trader_state = TraderState::delete_orders;
                    }
                    else if (bitmex_account->get_contracts() <= 0 && bitmex_account->get_mark_price() < action_take_profit) {
                        //position.market_order(action_leverage, event.price);
                        trader_state = TraderState::delete_orders;
                    }
                }
            }
            else {
                std::this_thread::sleep_for(20ms);
            }
        }
        /*
        else if (trader_state == TraderState::bitbot_action) {


            //const auto [leverage, stop_loss, take_profit] = mt_policy->get_action(agg_tick, bitmex_account->get_leverage());
            //const auto buy = rl_policy->get_action(current_interval_feature, bitmex_account->get_leverage());
            //if (place_order) {
            //if (buy) {
                //desired_leverage = action_leverage;
            if (direction_long) {
                desired_leverage = BitSim::BitMex::max_leverage;
            }
            else {
                desired_leverage = -BitSim::BitMex::max_leverage;
            }
            desired_ask_price = bitmex_account->get_ask_price();
            desired_bid_price = bitmex_account->get_bid_price();
            trader_state = TraderState::delete_orders;
            //}
            //else {
            //    trader_state = TraderState::wait_for_next_interval;
            //}
        }
        else if (trader_state == TraderState::delete_orders) {
            trader_state = TraderState::delete_orders_worker;
        }
        */
        else if (trader_state == TraderState::delete_orders) {
            const auto success = bitmex_rest_api->delete_all();
            if (success) {
                trader_state = TraderState::place_new_order;
            }
            else {
                if (system_clock_ms_now() - current_interval_timestamp > 2s) {
                    trader_state = TraderState::wait_for_next_agg_tick;
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
            const auto success = market_order();
            if (success) {
                std::this_thread::sleep_for(1s);
                trader_state = TraderState::order_monitoring;
            }
            else {
                if (system_clock_ms_now() - current_interval_timestamp > 5s) {
                    trader_state = TraderState::wait_for_next_agg_tick;
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
            if (system_clock_ms_now() - current_interval_timestamp > 5s) {
                trader_state = TraderState::wait_for_next_agg_tick;
            }
            else if (bitmex_account->count_orders() == 0) {
                trader_state = TraderState::wait_for_next_agg_tick;
            } 
            else if (bitmex_account->get_ask_price() != order_ask_price || bitmex_account->get_bid_price() != order_bid_price) {
                trader_state = TraderState::delete_orders;
            }
        }
    }
}

/*
bool BitmexTrader::limit_order(void)
{
    const auto position_contracts = bitmex_account->get_contracts();
    const auto mark_price = bitmex_account->get_mark_price();
    const auto wallet = bitmex_account->get_wallet();
    const auto upnl = bitmex_account->get_upnl();

    if (wallet == 0.0) {
        return true;
    }

    const auto max_contracts = BitSim::BitMex::max_leverage * (wallet + upnl) * mark_price;
    const auto margin = wallet * std::clamp(desired_leverage, -BitSim::BitMex::max_leverage, BitSim::BitMex::max_leverage);
    const auto desired_contracts = std::clamp(margin * mark_price, -max_contracts, max_contracts);
    const auto order_contracts = int(desired_contracts - position_contracts);

    if (order_contracts == 0) {
        return true;
    }

    order_bid_price = bitmex_account->get_bid_price();
    order_ask_price = bitmex_account->get_ask_price();
    const auto range = std::floor(mark_price / 10000) / 2 + 0.5;
    if (order_contracts > 0) {
        order_price = std::min(order_bid_price, mark_price + range);
    }
    else {
        order_price = std::max(order_ask_price, mark_price - range);
    }

    auto success = bitmex_rest_api->limit_order(order_contracts, order_price);

    //logger.info("Limit order s(%d) oc(%d) op(%0.1f)", success, order_contracts, order_price);

    return success;
}
*/

bool BitmexTrader::market_order(void)
{
    return true;
}
