#include "OrderManager.h"
#include "BitLib/Logger.h"

#include <functional>


OrderManager::OrderManager(sptrPortfolio portfolio, sptrOrderBooks order_books) :
    portfolio(portfolio), order_books(order_books), order_manager_running(true), state_side(StateSide::buying), state_order(StateOrder::place_order), order_manager_thread_running(true)
{
    state_timeout = DateTime::now() + 5s;

    //order_manager_thread = std::make_unique<std::thread>(&OrderManager::order_manager_worker, this);
}

void OrderManager::set_callbacks(place_order_t place_order_f, cancel_order_t cancel_order_f, replace_order_t replace_order_f)
{
    place_order = place_order_f;
    cancel_order = cancel_order_f;
    replace_order = replace_order_f;
}

void OrderManager::order_manager_worker(void)
{

    //logger.info("tick");

    /*
    static const auto order_size = 0.001;
    static const auto symbol = string_to_symbol("BTCUSDT");

    //state_timeout = DateTime::now() + 5s;
    //while (order_manager_running) {
        if (state_timeout > DateTime::now()) {
            //std::this_thread::sleep_for(100ms);
            return;
        }

        if (state_side == StateSide::starting) {
            state_side = StateSide::buying;
            state_order = StateOrder::place_order;
        }
        else if (state_side == StateSide::endless_loop) {
            //std::this_thread::sleep_for(100ms);
            return;
        }
        else {
            if (state_order == StateOrder::place_order) {
                const auto ask = (*order_books)[symbol.idx].get_last_ask();
                const auto bid = (*order_books)[symbol.idx].get_last_bid();
                auto price = 0.0;
                if (ask - bid == 0.5) {
                    price = state_side == StateSide::buying ? bid : ask;
                }
                else if (ask - bid < 2.0) {
                    price = state_side == StateSide::buying ? bid + 0.5 : ask - 0.5;
                }
                else {
                    price = state_side == StateSide::buying ? bid + 0.5 : ask - 0.5;
                }

                //const auto price = state_side == StateSide::buying ? (*order_books)[symbol.idx].get_last_bid() : (*order_books)[symbol.idx].get_last_ask();                
                const auto size = state_side == StateSide::buying ? order_size : -portfolio->positions_buy[symbol.idx].qty;
                const auto side = state_side == StateSide::buying ? Side::buy : Side::sell;
                logger.info("Place order %.4f @ %.2f", size, price);
                place_order(symbol, side, size, price);
                state_order = StateOrder::wait_until_fulfilled;
            }
            else if (state_order == StateOrder::wait_until_fulfilled) {
                if (portfolio->orders[symbol.idx].size() == 0) {
                    if ((state_side == StateSide::buying && portfolio->positions_buy[symbol.idx].qty == order_size) ||
                        (state_side == StateSide::selling && portfolio->positions_sell[symbol.idx].qty == order_size)) {
                        logger.info("Order fulfilled");
                        state_side = state_side == StateSide::buying ? StateSide::selling : StateSide::buying;
                        state_order = StateOrder::place_order;
                    }
                    else {
                        logger.info("Order fulfilled but position uncertain");
                        for (auto i = 0; i < 10; i++) {
                            std::this_thread::sleep_for(100ms);
                            if ((state_side == StateSide::buying && portfolio->positions_buy[symbol.idx].qty == order_size) ||
                                (state_side == StateSide::selling && portfolio->positions_buy[symbol.idx].qty == 0)) {
                                state_side = state_side == StateSide::buying ? StateSide::selling : StateSide::buying;
                                state_order = StateOrder::place_order;
                                break;
                            }
                        }
                        if (state_order != StateOrder::place_order) {
                            // Error
                            state_side = state_side == StateSide::buying ? StateSide::selling : StateSide::buying;
                            state_order = StateOrder::place_order;
                        }
                    }
                }
                else {
                    const auto ask = (*order_books)[symbol.idx].get_last_ask();
                    const auto bid = (*order_books)[symbol.idx].get_last_bid();
                    auto new_price = 0.0;
                    if (ask - bid == 0.5) {
                        new_price = state_side == StateSide::buying ? bid : ask;
                    }
                    else if (ask - bid < 2.0) {
                        new_price = state_side == StateSide::buying ? bid + 0.5 : ask - 0.5;
                    }
                    else {
                        new_price = state_side == StateSide::buying ? bid + 0.5 : ask - 0.5;
                    }

                    //const auto price = state_side == StateSide::buying ? (*order_books)[symbol.idx].get_last_bid() : (*order_books)[symbol.idx].get_last_ask();
                    const auto order = portfolio->orders[symbol.idx].front();
                    if ((state_side == StateSide::buying && new_price > order.price) ||
                        (state_side == StateSide::selling && new_price < order.price)) {
                        //auto new_price = state_side == StateSide::buying ? price - 0.5 : price + 0.5;
                        logger.info("Buying order price changed, replace from %.1f to %.1f", order.price, new_price);
                        replace_order(symbol, order.id, 0, new_price);
                        state_order = StateOrder::wait_until_replaced;
                        state_timeout = DateTime::now() + 2s;
                    }
                    else {
                        // TODO: timeout?
                        //std::this_thread::sleep_for(1us);
                    }
                }
            }
            else if (state_order == StateOrder::wait_until_replaced) {
                if (portfolio->orders[symbol.idx].size() == 0) {
                    state_side = state_side == StateSide::buying ? StateSide::selling : StateSide::buying;
                    state_order = StateOrder::place_order;
                }
                else {
                    const auto order = portfolio->orders[symbol.idx].front();
                    if (!order.replacing) {
                        logger.info("Buying order replaced");
                        state_order = StateOrder::wait_until_fulfilled;
                    }
                }
            }
            else if (state_order == StateOrder::wait_until_canceled) {
                if (portfolio->orders[symbol.idx].size() == 0) { // || (!portfolio->orders[symbol.idx].front().confirmed && portfolio->orders[symbol.idx].front().created + 100ms > DateTime::now())) {
                    state_order = StateOrder::place_order;
                    logger.info("Buying order canceled");
                }
                else {
                    state_timeout = DateTime::now() + 100ms;
                    //std::this_thread::sleep_for(10ms);
                }
            }
        }
    //}
    */
}

void OrderManager::order_book_updated(void)
{
    const std::lock_guard<std::mutex> lock(order_mutex);

    static const auto order_size = 0.001;
    static const auto symbol = string_to_symbol("BTCUSDT");

    const auto updated = (*order_books)[symbol.idx].updated();

    if (state_timeout > DateTime::now()) {
        return;
    }

    if (state_order == StateOrder::wait_until_replaced) {
        if (portfolio->orders[symbol.idx].size() == 1) {
            const auto& order = portfolio->orders[symbol.idx].front();
            if (order.replacing_qty == 0.0 && order.replacing_price == 0.0) {
                state_order = StateOrder::wait_until_fulfilled;
            }
        }
        else {
            logger.error("wait until replaced error");
            state_order = StateOrder::error;
        }
    }

    if (updated) {
        if (state_order == StateOrder::place_order) {
            const auto ask = (*order_books)[symbol.idx].get_last_ask();
            const auto bid = (*order_books)[symbol.idx].get_last_bid();

            if (ask - bid < 20.0) {
                const auto price = state_side == StateSide::buying ? ask - 0.5 : bid + 0.5;
                const auto size = state_side == StateSide::buying ? order_size : -portfolio->positions_buy[symbol.idx].qty;
                const auto side = state_side == StateSide::buying ? Side::buy : Side::sell;
                place_order(symbol, side, size, price);
                //logger.info("Place order %.3f @ %.2f", size, price);
                state_order = StateOrder::wait_until_fulfilled;
            }
        }
        else if (state_order == StateOrder::wait_until_fulfilled) {
            if (portfolio->orders[symbol.idx].size() == 1) {
                const auto ask = (*order_books)[symbol.idx].get_last_ask();
                const auto bid = (*order_books)[symbol.idx].get_last_bid();
                const auto id = portfolio->orders[symbol.idx].front().id;

                if (ask - bid > 20.0) {
                    cancel_order(symbol, id);
                    logger.info("Cancel order %s", id.to_string().c_str());
                    state_order = StateOrder::wait_until_canceled;
                }
                else {
                    const auto& order = portfolio->orders[symbol.idx].front();
                    const auto price = state_side == StateSide::buying ? ask - 0.5 : bid + 0.5;
                    if ((state_side == StateSide::buying && price > order.price) ||
                        (state_side == StateSide::selling && price < order.price)) {
                        auto size = state_side == StateSide::buying ? order_size : -portfolio->positions_buy[symbol.idx].qty;
                        if (size == order.qty) {
                            size = 0.0;
                        }
                        if (order.replacing_price == price) {
                            logger.info("Not replacing order, same price %.3f %.2f", size, price);
                        }
                        else {
                            replace_order(symbol, id, size, price);
                            logger.info("Replace order %.3f %.2f", size, price);
                        }
                        //state_order = StateOrder::wait_until_replaced;
                    }
                }
            }
            else if (portfolio->orders[symbol.idx].size() > 1) {
                state_order = StateOrder::error;
                logger.error("wait until fulfilled, too many orders");
            }
            else {
                state_order = StateOrder::place_order;
            }
        }
    }

    const auto& asks = (*order_books)[symbol.idx].asks;
    const auto& bids = (*order_books)[symbol.idx].bids;
    if (updated) {
        logger.info("OrderBook %.1f (%03.3f)   %.1f (%03.3f)   %.1f (%03.3f)  +  %.1f (%03.3f)   %.1f (%03.3f)   %.1f (%03.3f)", asks[2]->price, asks[2]->qty, asks[1]->price, asks[1]->qty, asks[0]->price, asks[0]->qty, bids[0]->price, bids[0]->qty, bids[1]->price, bids[1]->qty, bids[2]->price, bids[2]->qty);
    }
    else {
        logger.info("OrderBook %.1f (%03.3f)   %.1f (%03.3f)   %.1f (%03.3f)  -  %.1f (%03.3f)   %.1f (%03.3f)   %.1f (%03.3f)", asks[2]->price, asks[2]->qty, asks[1]->price, asks[1]->qty, asks[0]->price, asks[0]->qty, bids[0]->price, bids[0]->qty, bids[1]->price, bids[1]->qty, bids[2]->price, bids[2]->qty);
    }


    /*
    auto str = std::string{ "OrderBook:" };
    for (const auto& symbol : symbols) {
        str += " ";
        str += symbol.name;
        str += " (" + std::to_string((*order_books)[symbol.idx].get_last_ask()) + " " + std::to_string((*order_books)[symbol.idx].get_last_bid()) + ")";
    }
    */
    //logger.info(str.c_str());
}

void OrderManager::order_updated(void)
{
    const std::lock_guard<std::mutex> lock(order_mutex);

    static const auto symbol = string_to_symbol("BTCUSDT");
    if (state_order == StateOrder::wait_until_fulfilled) {
        if (portfolio->orders[symbol.idx].size() == 0) {
            //state_order = StateOrder::place_order;
        }
        else if (portfolio->orders[symbol.idx].front().confirmed) {

        }
        else {
            // Order unconfirmed and timed out?
        }
    }
    else if (state_order == StateOrder::wait_until_canceled) {
        if (portfolio->orders[symbol.idx].size() == 0) {
            state_order = StateOrder::place_order;
        }
    }
}

void OrderManager::portfolio_updated(void)
{
    const std::lock_guard<std::mutex> lock(order_mutex);

    portfolio->debug_print();
}
