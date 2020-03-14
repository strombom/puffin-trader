#include "pch.h"

#include "Utils.h"
#include "DateTime.h"
#include "BitmexSimulator.h"


void BitmexSimulator::reset(void)
{
    constexpr auto episode_length = ((std::chrono::seconds) BitSim::Closer::episode_length).count() / ((std::chrono::seconds) BitSim::interval).count();
    intervals_idx = 0; // Utils::random(0, (int)intervals->rows.size() - 1);
    intervals_idx_start = intervals_idx;
    intervals_idx_end = intervals_idx + episode_length;
    
    wallet = 1.0;
    pos_price = 0.0;
    pos_contracts = 0.0;
}

double BitmexSimulator::get_value(void)
{
    return 0.0;
}

double BitmexSimulator::sigmoid_to_price(double price, double sigmoid) {
    // High output when sigmoid near 0
    // Low output when sigmoid near 1
    auto p = std::pow(price / 100.0, 5.0 * (1.0 - sigmoid) / 4.0 - 0.25) - 1.0;
    p = std::round(p * 2.0) * 0.5;
    return std::max(0.0, p);
}

std::tuple<double, double> BitmexSimulator::calculate_order_size(double buy_size, double sell_size)
{
    if (wallet == 0.0) {
        return std::make_tuple(0.0, 0.0);
    }

    const auto mark_price = intervals->rows[intervals_idx].last_price;
    auto position_margin = 0.0;
    auto position_leverage = 0.0;
    auto upnl = 0.0;

    if (pos_contracts != 0.0) {
        const auto sign = pos_contracts / abs(pos_contracts);
        const auto entry_value = std::abs(pos_contracts / pos_price);
        const auto mark_value = std::abs(pos_contracts / mark_price);
        upnl = sign * (entry_value - mark_value);

        position_margin = std::max(0.0, std::abs(pos_contracts / pos_price) - upnl);
        position_leverage = position_margin / wallet;
    }

    const auto max_margin = BitSim::BitMex::max_leverage * wallet;
    const auto available_margin = max_margin - position_margin;
    const auto max_contracts = BitSim::BitMex::max_leverage * (wallet + upnl) * mark_price;

    auto max_buy_contracts = max_margin * mark_price;
    auto max_sell_contracts = max_margin * mark_price;

    if (pos_contracts > 0.0) {
        max_buy_contracts = std::max(0.0, available_margin * mark_price);
        max_sell_contracts = std::max(0.0, max_contracts + pos_contracts);
    }
    else if (pos_contracts < 0.0) {
        max_buy_contracts = std::max(0.0, max_contracts - pos_contracts);
        max_sell_contracts = std::max(0.0, available_margin * mark_price);
    }

    buy_size = std::max(0.0, buy_size - sell_size);
    sell_size = std::max(0.0, sell_size - buy_size);
    const auto buy_fraction = std::max(0.0, (buy_size - BitSim::BitMex::order_hysteresis) / (1.0 - BitSim::BitMex::order_hysteresis));
    const auto sell_fraction = std::max(0.0, (sell_size - BitSim::BitMex::order_hysteresis) / (1.0 - BitSim::BitMex::order_hysteresis));

    const auto buy_contracts = std::min(max_buy_contracts, max_contracts * buy_fraction);
    const auto sell_contracts = std::min(max_sell_contracts, max_contracts * sell_fraction);

    return std::make_tuple(buy_contracts, sell_contracts);
}

RL_State BitmexSimulator::step(const RL_Action& action)
{
    const auto prev_interval = intervals->rows[intervals_idx];

    const auto [buy_contracts, sell_contracts] = calculate_order_size(action.buy_size, action.sell_size);

    const auto buy_delta_price = sigmoid_to_price(prev_interval.last_price, action.buy_position);
    const auto sell_delta_price = sigmoid_to_price(prev_interval.last_price, action.sell_position);

    std::cout << "--- " << std::endl;
    std::cout << "action.buy_size: " << action.buy_size << std::endl;
    std::cout << "action.sell_size: " << action.sell_size << std::endl;
    std::cout << "buy_n_contracts: " << buy_contracts << std::endl;
    std::cout << "sell_n_contracts: " << sell_contracts << std::endl;

    std::cout << "--- " << std::endl;

    if (buy_contracts > 0) {
        if (buy_delta_price == 0) {
            std::cout << "Market buy" << std::endl;
            market_order(buy_contracts);

        }
        else {
            std::cout << "Limit buy" << std::endl;
            limit_order(buy_contracts, prev_interval.last_price - buy_delta_price);
        }
    }

    if (sell_contracts > 0) {
        if (sell_delta_price == 0) {
            std::cout << "Market sell" << std::endl;
            market_order(-sell_contracts);

        }
        else {
            std::cout << "Limit sell" << std::endl;
            limit_order(-sell_contracts, prev_interval.last_price + sell_delta_price);

        }
    }

    std::cout << "--- " << std::endl;
    
    auto state = RL_State{};
    state.set_done();
    
    ++intervals_idx;
    if (intervals_idx == intervals_idx_end - 1) {
        state.set_done();
    }

    return state;
}

void BitmexSimulator::market_order(double contracts)
{
    std::cout << "market_order size(" << contracts << ")" << std::endl;
    const auto next_price = intervals->rows[intervals_idx + 1].last_price;

    if (contracts > 0.0) {
        // Buy
        execute_order(contracts, next_price + 0.5, true);
    } 
    else if (contracts < 0.0) {
        // Sell
        execute_order(contracts, next_price - 0.5, true);

    }
}

void BitmexSimulator::limit_order(double contracts, double price)
{
    std::cout << "limit_order size(" << contracts << ") price(" << price << ")" << std::endl;
    const auto next_price = intervals->rows[intervals_idx + 1].last_price;

    if ((contracts > 0.0 && next_price < price) ||
        (contracts < 0.0 && next_price > price)) {
        execute_order(contracts, price, false);
    }
}

void BitmexSimulator::execute_order(double order_contracts, double price, bool taker)
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
        wallet += (1 / pos_price - 1 / price) * std::min(-order_contracts, pos_contracts);
    }
    else if (pos_contracts < 0 && order_contracts > 0) {
        wallet += (1 / pos_price - 1 / price) * std::max(-order_contracts, pos_contracts);
    }

    // Calculate average entry price
    if ((pos_contracts >= 0 && order_contracts > 0) || 
        (pos_contracts <= 0 && order_contracts < 0)) {
        pos_price = (pos_contracts * pos_price + order_contracts * price) / (pos_contracts + order_contracts);
    }
    else if ((pos_contracts >= 0 && (pos_contracts + order_contracts) < 0) || 
             (pos_contracts <= 0 && (pos_contracts + order_contracts) > 0)) {
        pos_price = price;
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

    const auto entry_value = pos_contracts / pos_price;
    const auto maintenance_margin = BitSim::BitMex::maintenance_rate * abs(entry_value);
    const auto liquidation_fee = BitSim::BitMex::taker_fee * abs(wallet + entry_value);

    auto liquidation_price = (pos_contracts * pos_price) / (pos_price * (wallet - maintenance_margin - liquidation_fee) + pos_contracts);
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
