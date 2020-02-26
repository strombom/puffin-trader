#include "pch.h"

#include "Utils.h"
#include "DateTime.h"
#include "BitmexSimulator.h"


void BitmexSimulator::reset(void)
{
    const auto timeout_length = ((std::chrono::seconds) BitSim::Closer::closing_timeout).count() / ((std::chrono::seconds) BitSim::interval).count();
    intervals_idx = Utils::random(0, intervals->rows.size() - timeout_length); 
    intervals_idx_start = intervals_idx;
    intervals_idx_end = intervals_idx + timeout_length;
    
    wallet = 0.0f;
    entry_price = 0.0f;
    pos_contracts = 0.0f;
}

void BitmexSimulator::put_order(float price, float contracts)
{
    const auto fill_price = price;
    execute_order(fill_price, contracts, true);
}

void BitmexSimulator::execute_order(float price, float order_contracts, bool taker)
{
    // Fees
    if (taker) {
        wallet -= BitSim::BitMex::taker_fee * fabs(order_contracts / price);
    }
    else {
        wallet -= BitSim::BitMex::maker_fee * fabs(order_contracts / price);
    }
    
    // Realised profit and loss
    if (pos_contracts > 0 && order_contracts < 0) {
        wallet += (1 / entry_price - 1 / price) * fmin(-order_contracts, pos_contracts);
    }
    else if (pos_contracts < 0 && order_contracts > 0) {
        wallet += (1 / entry_price - 1 / price) * fmax(-order_contracts, pos_contracts);
    }

    // Calculate average entry price
    if ((pos_contracts >= 0 && order_contracts > 0) || 
        (pos_contracts <= 0 && order_contracts < 0)) {
        entry_price = (pos_contracts * entry_price + order_contracts * price) / (pos_contracts + order_contracts);
    }
    else if ((pos_contracts >= 0 && (pos_contracts + order_contracts) < 0) || 
             (pos_contracts <= 0 && (pos_contracts + order_contracts) > 0)) {
        entry_price = price;
    }

    // Calculate position contracts
    pos_contracts += order_contracts;
}
