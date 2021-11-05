
#include "Portfolio.h"
#include "BitLib/Logger.h"


void Portfolio::update_order(Uuid id, const Symbol& symbol, Side side, double price, double qty, std::string status, time_point_us created)
{
    //logger.info("Update %s", id.to_string().c_str());
    if (status == "Filled" || status == "Cancelled" || status == "Rejected") {
        if (orders.contains(id)) {
            orders.erase(id);
        }
    }
    else {
        // Created, New, PartiallyFilled, PendingCancel
        orders.insert_or_assign(id, Order{id, symbol, qty, price, side, created});
    }

    for (const auto& order : orders) {
        logger.info("Order: %s %s %f %f", order.second.uuid.to_string().c_str(), order.second.symbol.name.data(), order.second.price, order.second.qty);
    }
}

void Portfolio::update_position(const Symbol& symbol, Side side, double qty)
{
    if (side == Side::buy) {
        positions_buy[symbol.idx].qty = qty;
    }
    else {
        positions_sell[symbol.idx].qty = qty;
    }
    for (const auto& symbol : symbols) {
        if (positions_buy[symbol.idx].qty > 0 && positions_sell[symbol.idx].qty) {
            logger.info("Position: %s buy(%.5f) sell(%.5f)", symbol.name.data(), positions_buy[symbol.idx].qty, positions_sell[symbol.idx].qty);
        }
        else if (positions_buy[symbol.idx].qty > 0) {
            logger.info("Position: %s buy(%.5f)", symbol.name.data(), positions_buy[symbol.idx].qty);
        }
        else if (positions_sell[symbol.idx].qty) {
            logger.info("Position: %s sell(%.5f)", symbol.name.data(), positions_sell[symbol.idx].qty);
        }
    }
}

void Portfolio::update_wallet(double balance, double available)
{
    wallet_balance = balance;
    wallet_available = available;
    logger.info("Wallet balance: %.5f available: %.5f", wallet_balance, wallet_available);
}

void Portfolio::new_trade(const Symbol& symbol, Side side, double price)
{
    const auto bid_price = side == Side::buy ? price - symbol.tick_size : price;

    if (bid_price != last_bid[symbol.idx]) {
        last_bid[symbol.idx] = bid_price;
        logger.info("New bid price %s %.5f", symbol.name.data(), bid_price);
    }
    if (side == Side::buy) {
        logger.info("Trade: %s Buy %.5f", symbol.name.data(), price);
    }
    else {
        logger.info("Trade: %s Sell %.5f", symbol.name.data(), price);
    }
}

void Portfolio::order_book_clear(const Symbol& symbol)
{
    order_books[symbol.idx].clear();
}

void Portfolio::order_book_insert(const Symbol& symbol, double price, Side side, double qty)
{
    order_books[symbol.idx].insert(price, side, qty);
}

void Portfolio::order_book_update(const Symbol& symbol, double price, Side side, double qty)
{
    order_books[symbol.idx].update(price, side, qty);
}

void Portfolio::order_book_delete(const Symbol& symbol, double price, Side side)
{
    order_books[symbol.idx].del(price, side);
}

double Portfolio::order_book_get_last_bid(const Symbol& symbol)
{
    return order_books[symbol.idx].get_last_bid();
}

OrderBook::OrderBook(void)
{
    for (auto idx = 0; idx < size; idx++) {
        asks[idx] = new Entry();
        bids[idx] = new Entry();
    }
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
    //auto entries = side == Side::buy ? &bids : &asks;

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
    return bids[0]->price;
}
