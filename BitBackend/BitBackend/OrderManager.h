#pragma once
#include "precompiled_headers.h"

#include "OrderBook.h"
#include "Portfolio.h"
#include "BitLib/DateTime.h"


class OrderManager
{
private:
    using place_order_t = std::function<void(const Symbol& symbol, Side side, double qty, double price)>;
    using cancel_order_t = std::function<void(const Symbol& symbol, Uuid id_external)>;
    using replace_order_t = std::function<void(const Symbol& symbol, Uuid id_external, double qty, double price)>;

public:
    OrderManager(sptrPortfolio portfolio, sptrOrderBooks order_books);

    void set_callbacks(place_order_t place_order_f, cancel_order_t cancel_order_f, replace_order_t replace_order_f);

    void order_book_updated(void);
    void order_updated(void);
    void position_updated(void);

    enum class StateSide {
        buying,
        selling
    };

    enum class StateOrder {
        place_order,
        wait_until_fulfilled,
        wait_until_canceled,
        wait_until_replaced,
        error
    };

    sptrPortfolio portfolio;
    sptrOrderBooks order_books;

private:
    std::mutex order_mutex;

    bool order_manager_running;
    StateSide state_side;
    StateOrder state_order;
    time_point_us state_timeout;
    
    std::atomic_bool order_manager_thread_running;
    std::unique_ptr<std::thread> order_manager_thread;

    place_order_t place_order;
    cancel_order_t cancel_order;
    replace_order_t replace_order;

    void order_manager_worker(void);
};

using sptrOrderManager = std::shared_ptr<OrderManager>;
