#include "OrderManager.h"
#include "BitLib/Logger.h"

#include <functional>


OrderManager::OrderManager(sptrPortfolio portfolio, sptrOrderBooks order_books) :
    portfolio(portfolio), order_books(order_books), order_manager_running(true), state_side(StateSide::starting), state_order(StateOrder::place_order), order_manager_thread_running(true)
{
    order_manager_thread = std::make_unique<std::thread>(&OrderManager::order_manager_worker, this);
}

void OrderManager::set_callbacks(place_order_t place_order_f, cancel_order_t cancel_order_f)
{
    place_order = place_order_f;
    cancel_order = cancel_order_f;
}

void OrderManager::order_manager_worker(void)
{
    const auto order_size = 0.001;
    const auto symbol = string_to_symbol("BTCUSDT");

    state_timeout = DateTime::now() + 5s;
    while (order_manager_running) {
        if (state_side == StateSide::starting) {
            if (DateTime::now() > state_timeout) {
                state_side = StateSide::buying;
                state_order = StateOrder::place_order;
            }
            else {
                std::this_thread::sleep_for(100ms);
            }
        }
        else if (state_side == StateSide::endless_loop) {
            std::this_thread::sleep_for(100ms);
        }
        else {
            if (state_order == StateOrder::place_order) {
                const auto price = state_side == StateSide::buying ? (*order_books)[symbol.idx].get_last_ask() - 0.5 : (*order_books)[symbol.idx].get_last_bid() + 0.5;
                const auto size = state_side == StateSide::buying ? order_size : -order_size;
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
                    const auto price = state_side == StateSide::buying ? (*order_books)[symbol.idx].get_last_ask() : (*order_books)[symbol.idx].get_last_bid();
                    const auto order = portfolio->orders[symbol.idx].front();
                    if ((state_side == StateSide::buying && price > order.price + 0.5) ||
                        (state_side == StateSide::selling && price < order.price - 0.5)) {
                        logger.info("Buying order price changed, cancel");
                        cancel_order(symbol, order.id);
                        state_order = StateOrder::wait_until_canceled;
                    }
                    else {
                        // TODO: timeout?
                        std::this_thread::sleep_for(1ms);
                    }
                }
            }
            else if (state_order == StateOrder::wait_until_canceled) {
                if (portfolio->orders[symbol.idx].size() == 0) { // || (!portfolio->orders[symbol.idx].front().confirmed && portfolio->orders[symbol.idx].front().created + 100ms > DateTime::now())) {
                    state_order = StateOrder::place_order;
                    logger.info("Buying order canceled");
                }
                else {
                    std::this_thread::sleep_for(10ms);
                }
            }
        }
    }
}

void OrderManager::order_book_updated(void)
{
    auto str = std::string{ "OrderBook:" };
    for (const auto& symbol : symbols) {
        str += " ";
        str += symbol.name;
        str += " (" + std::to_string((*order_books)[symbol.idx].get_last_ask()) + " " + std::to_string((*order_books)[symbol.idx].get_last_bid()) + ")";
    }
    //logger.info(str.c_str());
}

void OrderManager::portfolio_updated(void)
{
    portfolio->debug_print();
}
