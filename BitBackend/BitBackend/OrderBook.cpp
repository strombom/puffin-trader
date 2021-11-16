
#include "BitLib/Logger.h"
#include "OrderBook.h"


OrderBook::OrderBook(void)
{
    for (auto idx = 0; idx < size; idx++) {
        asks[idx] = new Entry();
        bids[idx] = new Entry();
    }
    old_last_bid = 0.0;
}

void OrderBook::clear(void)
{
    for (auto idx = 0; idx < size; idx++) {
        asks[idx]->price = 0.0;
        bids[idx]->price = 0.0;
    }
}

void OrderBook::insert(double price, Side side, double qty)
{
    if (side == Side::buy) {
        auto insert_idx = 0;
        for (insert_idx = 0; insert_idx < size; insert_idx++) {
            if (bids[insert_idx]->price == 0) {
                break;
            }
            if (bids[insert_idx]->price == price) {
                logger.warn("Insert bid error");
            }
            if (bids[insert_idx]->price < price) {
                break;
            }
        }
        for (auto idx = size - 1; idx > insert_idx; idx--) {
            std::swap(bids[idx], bids[idx - 1]);
        }
        bids[insert_idx]->price = price;
        bids[insert_idx]->qty = qty;
    }
}

void OrderBook::update(double price, Side side, double qty)
{
    if (side == Side::buy) {
        for (auto idx = 0; idx < size; idx++) {
            if (bids[idx]->price == 0) {
                logger.warn("Update bid error 1");
                break;
            }
            if (bids[idx]->price == price) {
                bids[idx]->qty = qty;
                break;
            }
            if (bids[idx]->price < price) {
                logger.warn("Update bid error 2");
                break;
            }
        }
    }
}

void OrderBook::del(double price, Side side)
{
    if (side == Side::buy) {
        auto del_idx = 0;
        for (del_idx = 0; del_idx < size; del_idx++) {
            if (bids[del_idx]->price == 0) {
                logger.warn("Delete bid error 1");
                break;
            }
            if (bids[del_idx]->price == price) {
                bids[del_idx]->price = 0.0;
                break;
            }
            if (bids[del_idx]->price < price) {
                logger.warn("Update bid error 2");
                break;
            }
        }
        for (auto idx = del_idx; idx < size - 1; idx++) {
            std::swap(bids[idx], bids[idx + 1]);
        }
    }
}

double OrderBook::get_last_bid(void)
{
    if (old_last_bid != bids[0]->price) {
        old_last_bid = bids[0]->price;
        logger.info("Bid %.2f", old_last_bid);
    }

    return bids[0]->price;
}

sptrOrderBooks makeOrderBooks(void) {
    return std::make_shared< std::array<OrderBook, symbols.size()>>();
}
