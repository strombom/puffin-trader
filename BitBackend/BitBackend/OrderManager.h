#pragma once
#include "precompiled_headers.h"

#include "OrderBook.h"
#include "Portfolio.h"


class OrderManager
{
public:
    OrderManager(sptrPortfolio portfolio, sptrOrderBooks order_books);

    void order_book_updated(void);
    void portfolio_updated(void);

    enum class State {
        buying,
        selling
    };

    sptrPortfolio portfolio;
    sptrOrderBooks order_books;

private:
    State state;

    std::atomic_bool order_manager_thread_running;
    std::unique_ptr<std::thread> order_manager_thread;

    void order_manager_worker(void);
};

using sptrOrderManager = std::shared_ptr<OrderManager>;
