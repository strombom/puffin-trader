#include "OrderManager.h"
#include "BitLib/Logger.h"

#include <functional>


OrderManager::OrderManager(sptrPortfolio portfolio, sptrOrderBooks order_books) :
    portfolio(portfolio), order_books(order_books), state(State::buying), order_manager_thread_running(true)
{
    order_manager_thread = std::make_unique<std::thread>(&OrderManager::order_manager_worker, this);
}

void OrderManager::order_manager_worker(void)
{

}

void OrderManager::order_book_updated(void)
{
    logger.info("order book updated");
}

void OrderManager::portfolio_updated(void)
{
    logger.info("portfolio updated");
}
