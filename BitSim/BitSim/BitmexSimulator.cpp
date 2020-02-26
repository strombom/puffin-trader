#include "pch.h"

#include "Utils.h"
#include "DateTime.h"
#include "BitmexSimulator.h"


void BitmexSimulator::reset(void)
{
    const auto timeout_length = ((std::chrono::seconds) BitSim::Closer::closing_timeout).count() / ((std::chrono::seconds) BitSim::interval).count();
    intervals_idx = Utils::random(0, (int) intervals->rows.size() - timeout_length); 
    intervals_idx_start = intervals_idx;
    intervals_idx_end = intervals_idx + timeout_length;
    
    wallet = 0.0;
    entry_price = 0.0;
    pos_contracts = 0.0;
}

void BitmexSimulator::put_order(double price, double contracts)
{
    const auto fill_price = price;
    execute_order(fill_price, contracts, true);
}

void BitmexSimulator::execute_order(double price, double order_contracts, bool taker)
{
    // Fees
    if (taker) {
        wallet -= BitSim::BitMex::taker_fee * abs(order_contracts / price);
    }
    else {
        wallet -= BitSim::BitMex::maker_fee * abs(order_contracts / price);
    }
    
    // Realised profit and loss
    if (pos_contracts > 0 && order_contracts < 0) {
        wallet += (1 / entry_price - 1 / price) * std::min(-order_contracts, pos_contracts);
    }
    else if (pos_contracts < 0 && order_contracts > 0) {
        wallet += (1 / entry_price - 1 / price) * std::max(-order_contracts, pos_contracts);
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

double BitmexSimulator::liquidation_price(void)
{
    constexpr auto max_liquidation_price = 100000000.0;

    if (pos_contracts == 0) {
        return max_liquidation_price;
    }

    const auto entry_value = pos_contracts / entry_price;
    const auto maintenance_margin = BitSim::BitMex::maintenance_rate * abs(entry_value);
    const auto liquidation_fee = BitSim::BitMex::taker_fee * abs(wallet + entry_value);

    auto liquidation_price = (pos_contracts * entry_price) / (entry_price * (wallet - maintenance_margin - liquidation_fee) + pos_contracts);
    if (liquidation_price < 0) {
        if (pos_contracts < 0) {
            liquidation_price = max_liquidation_price;
        }
        else {
            liquidation_price = 0.0;
        }
    }

    return liquidation_price;
}
