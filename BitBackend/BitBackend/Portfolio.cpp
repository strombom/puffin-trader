
#include "Portfolio.h"
#include "BitLib/Logger.h"


Portfolio::Order* Portfolio::find_order(const Symbol& symbol, Uuid id)
{
    for (auto& order : orders[symbol.idx]) {
        if (order.id == id) {
            return &order;
        }
    }
    return nullptr;
}

void Portfolio::update_order(Uuid id, const Symbol& symbol, Side side, double qty, double price, time_point_us created, bool confirmed)
{
    const std::lock_guard<std::mutex> lock(orders_mutex);

    logger.info("update_order id(%s) %s, %.5f, %.2f", id.to_string().c_str(), symbol.name.data(), price, qty);

    auto order = find_order(symbol, id);

    logger.info("find order %d", order);

    if (order == nullptr && qty > 0) {
        orders[symbol.idx].emplace_front(id, symbol, side, qty, price, created, confirmed);
    }
    else if (order != nullptr && qty == 0) {
        orders[symbol.idx].remove(*order);
    }
    else if (order != nullptr) {
        *order = Order{ id, symbol, side, qty, price, created, confirmed };
    }
}

void Portfolio::update_position(const Symbol& symbol, Side side, double qty)
{
    const std::lock_guard<std::mutex> lock(positions_mutex);
    if (side == Side::buy) {
        positions_buy[symbol.idx].qty = qty;
    }
    else {
        positions_sell[symbol.idx].qty = qty;
    }
    /*
    for (const auto& symbol : symbols) {
        if (positions_buy[symbol.idx].qty > 0 && positions_sell[symbol.idx].qty > 0) {
            logger.info("Position: %s buy(%.5f) sell(%.5f)", symbol.name.data(), positions_buy[symbol.idx].qty, positions_sell[symbol.idx].qty);
        }
        else if (positions_buy[symbol.idx].qty > 0) {
            logger.info("Position: %s buy(%.5f)", symbol.name.data(), positions_buy[symbol.idx].qty);
        }
        else if (positions_sell[symbol.idx].qty > 0) {
            logger.info("Position: %s sell(%.5f)", symbol.name.data(), positions_sell[symbol.idx].qty);
        }
    }
    */
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

void Portfolio::debug_print(void)
{
    const std::lock_guard<std::mutex> lock(positions_mutex);

    auto str = std::string{"Portfolio:"};
    for (const auto& symbol : symbols) {
        str += " ";
        str += symbol.name;
        str += " (" + std::to_string(positions_buy[symbol.idx].qty) + " " + std::to_string(positions_sell[symbol.idx].qty) + ")";
    }
    logger.info(str.c_str());
}
