
#include "Portfolio.h"
#include "BitLib/Logger.h"


void Portfolio::update_order(Uuid id, const Symbol& symbol, Portfolio::Side side, double price, double qty, std::string status, time_point_us created)
{
    //logger.info("Update %s\n", id.to_string().c_str());
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
        logger.info("Order: %s %s %f %f\n", order.second.uuid.to_string().c_str(), order.second.symbol.name.data(), order.second.price, order.second.qty);
    }
}

void Portfolio::update_position(const Symbol& symbol, Portfolio::Side side, double qty)
{
    if (side == Portfolio::Side::buy) {
        positions_buy[symbol.idx].qty = qty;
    }
    else {
        positions_sell[symbol.idx].qty = qty;
    }
    for (const auto& symbol : symbols) {
        if (positions_buy[symbol.idx].qty > 0 && positions_sell[symbol.idx].qty) {
            logger.info("Position: %s buy(%.5f) sell(%.5f)\n", symbol.name.data(), positions_buy[symbol.idx].qty, positions_sell[symbol.idx].qty);
        }
        else if (positions_buy[symbol.idx].qty > 0) {
            logger.info("Position: %s buy(%.5f) \n", symbol.name.data(), positions_buy[symbol.idx].qty);
        }
        else if (positions_sell[symbol.idx].qty) {
            logger.info("Position: %s sell(%.5f)\n", symbol.name.data(), positions_sell[symbol.idx].qty);
        }
    }
}

void Portfolio::update_wallet(double balance, double available)
{
    wallet_balance = balance;
    wallet_available = available;
    logger.info("Wallet balance: %.5f available: %.5f\n", wallet_balance, wallet_available);
}
