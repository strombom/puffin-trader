#include "pch.h"
#include "MT_Simulator.h"


MT_Simulator::MT_Simulator(void) :
    wallet(0.0), pos_price(0.0), pos_contracts(0.0),
    time_since_leverage_change(0.0),
    orderbook_last_price(0.0)
{

}

void MT_Simulator::step(const Tick& tick)
{

}

void MT_Simulator::market_order(double contracts)
{

}

void MT_Simulator::limit_order(double contracts, double price)
{

}

void MT_Simulator::execute_order(double contracts, double price, double fee)
{

}

double MT_Simulator::calculate_order_size(double leverage)
{
    return 0.0;
}


/*
sptrRL_State MT_Simulator::reset(int idx_episode, bool validation, double _training_progress)
{
    training_progress = _training_progress;

    if (validation) {
        logger = std::make_unique<BitmexSimulatorLogger>("bitmex_val_" + std::to_string(idx_episode) + ".csv", true);
    }
    else {
        logger = std::make_unique<BitmexSimulatorLogger>("bitmex_train_" + std::to_string(idx_episode) + ".csv", true);
    }

    constexpr auto episode_length = (int)(((std::chrono::milliseconds) BitSim::Trader::episode_length).count() / ((std::chrono::milliseconds) BitSim::interval).count());

    const auto training_start_idx = BitSim::FeatureEncoder::observation_length - 1;
    const auto validation_end_idx = (int)intervals->rows.size() - episode_length;

    const auto training_end_idx = (int)((validation_end_idx - training_start_idx) * 4.0 / 5.0) - 1;
    const auto validation_start_idx = training_end_idx + 1;

    if (validation) {
        intervals_idx = Utils::random(validation_start_idx, validation_end_idx);
    }
    else {
        intervals_idx = Utils::random(training_start_idx, training_end_idx);
    }

    intervals_idx_start = intervals_idx;
    intervals_idx_end = intervals_idx + episode_length;

    wallet = 1.0;
    pos_price = 0.0;
    pos_contracts = 0.0;
    start_value = wallet * intervals->rows[intervals_idx_start].last_price;
    get_reward_previous_value = 0.0;

    orderbook_last_price = start_value;

    // Random leverage at start
    //const auto start_leverage = Utils::random(-BitSim::BitMex::max_leverage, BitSim::BitMex::max_leverage);
    const auto start_leverage = Utils::random_choice({ -BitSim::BitMex::max_leverage, BitSim::BitMex::max_leverage });
    market_order(calculate_order_size(start_leverage), false);

    constexpr auto position_reward = 0.0;
    constexpr auto time_since_change = 0.0;
    constexpr auto delta_price = 0.0;

    //std::cout << "Features " << features.sizes() << std::endl;
    //const auto feature = features[intervals_idx - (BitSim::FeatureEncoder::observation_length - 1)][0];
    const auto feature = features[intervals_idx - (BitSim::FeatureEncoder::observation_length - 1)];

    auto [_position_margin, position_leverage, upnl] = calculate_position_leverage(intervals->rows[intervals_idx].last_price);
    auto state = std::make_shared<RL_State>(position_reward, feature, position_leverage, delta_price, time_since_change);
    return state;
}

time_point_ms BitmexSimulator::get_start_timestamp(void)
{
    return intervals->get_timestamp_start() + BitSim::interval * intervals_idx_start;
}

sptrRL_State BitmexSimulator::step(sptrRL_Action action, bool last_step)
{
    const auto prev_interval = intervals->rows[intervals_idx];

    if (prev_interval.last_price > orderbook_last_price + 0.5) {
        orderbook_last_price = prev_interval.last_price - 0.5;
    }
    else if (prev_interval.last_price < orderbook_last_price) {
        orderbook_last_price = prev_interval.last_price;
    }

    //action->leverage;
    //action->limit_order;
    //action->market_order;

    const auto order_leverage = action->buy ? BitSim::BitMex::max_leverage : -BitSim::BitMex::max_leverage;

    if (order_leverage > 0 && pos_contracts < 0 ||
        order_leverage < 0 && pos_contracts > 0) {
        time_since_leverage_change = 0.0;
        const auto order_contracts = calculate_order_size(order_leverage);
        market_order(order_contracts, prev_interval.last_price);
    }
    else {
        time_since_leverage_change += 1;
    }

    //if (!action->idle) {
    //}

    //order_contracts = calculate_order_size(action->leverage);
    //limit_order(order_contracts, prev_interval.last_price);

    const auto time_since_change = std::log1p(time_since_leverage_change) / 5.0;

    auto delta_price = pos_price - prev_interval.last_price;
    const auto delta_price_sign = delta_price < 0 ? -1.0 : 1.0;
    delta_price = delta_price_sign * std::log1p(delta_price_sign * delta_price) / 3.0;

    //const auto feature = features[intervals_idx - (BitSim::FeatureEncoder::observation_length - 1)][0];
    const auto feature = features[intervals_idx - (BitSim::FeatureEncoder::observation_length - 1)];

    const auto reward = get_reward();
    auto [_position_margin, position_leverage, upnl] = calculate_position_leverage(prev_interval.last_price);
    auto state = std::make_shared<RL_State>(reward, feature, position_leverage, delta_price, time_since_change);

    logger->log(prev_interval.last_price,
        wallet,
        upnl,
        pos_contracts,
        position_leverage,
        action->buy,
        order_leverage,
        //order_contracts,
        //action->leverage,
        //action->idle,
        //action->limit_order,
        //action->market_order,
        reward);

    if (is_liquidated()) {
        wallet = 0.0;
        pos_price = 0.0;
        pos_contracts = 0.0;
        state->set_done();
    }

    ++intervals_idx;
    if (intervals_idx == intervals_idx_end - 1) {
        state->set_done();
    }

    return state;
}

double BitmexSimulator::get_reward(void)
{
    const auto next_price = intervals->rows[(long)(intervals_idx + 1)].last_price;
    auto position_pnl = 0.0;
    if (pos_contracts != 0) {
        //const auto exit_fee = BitSim::BitMex::taker_fee * abs(pos_contracts / next_price);
        const auto exit_fee = 0.0;
        position_pnl = pos_contracts * (1 / pos_price - 1 / next_price) - exit_fee;
    }
    const auto value = (wallet + position_pnl) * next_price / start_value;
    if (get_reward_previous_value == 0.0) {
        get_reward_previous_value = value;
    }
    //const auto reward = std::log(value / get_reward_previous_value) * 1000 - 1.0; // (*1000-1 to get a suitable reward range, between -1000 and -300)
    const auto reward = (value - get_reward_previous_value) * 10000.0 - 0.01;
    get_reward_previous_value = value;

    //std::cout.precision(3);
    //std::cout << std::fixed << "PNL(" << position_pnl << ")" << " value(" << value << ")" << " reward(" << reward << ")" << " previous_value(" << previous_value << ")" << " next_price(" << next_price << ")" << " pos_price(" << pos_price << ")" << std::endl;

    return reward;
}

std::tuple<double, double, double> BitmexSimulator::calculate_position_leverage(double mark_price)
{
    auto position_margin = 0.0;
    auto position_leverage = 0.0;
    auto upnl = 0.0;

    if (pos_contracts != 0.0) {
        const auto sign = pos_contracts / abs(pos_contracts);
        const auto entry_value = std::abs(pos_contracts / pos_price);
        const auto mark_value = std::abs(pos_contracts / mark_price);

        upnl = sign * (entry_value - mark_value);
        position_margin = std::max(0.0, std::abs(pos_contracts / pos_price) - upnl);
        position_leverage = sign * position_margin / wallet;
    }

    return std::make_tuple(position_margin, position_leverage, upnl);
}

double BitmexSimulator::calculate_order_size(double leverage)
{
    if (wallet == 0.0) {
        return 0.0;
    }

    const auto mark_price = intervals->rows[intervals_idx].last_price;
    const auto [position_margin, _position_leverage, upnl] = calculate_position_leverage(mark_price);

    const auto max_contracts = BitSim::BitMex::max_leverage * (wallet + upnl) * mark_price;
    const auto margin = wallet * std::clamp(leverage, -BitSim::BitMex::max_leverage, BitSim::BitMex::max_leverage);
    const auto contracts = std::clamp(margin * mark_price, -max_contracts, max_contracts);

    const auto order_contracts = contracts - pos_contracts;
    return order_contracts;
}

void BitmexSimulator::market_order(double contracts)
{
    market_order(contracts, true);
}

void BitmexSimulator::market_order(double contracts, bool use_fee)
{
    //std::cout << "market_order size(" << contracts << ")" << std::endl;
    const auto next_price = intervals->rows[(long)(intervals_idx + 1)].last_price;
    const auto fee = use_fee ? BitSim::BitMex::taker_fee * (training_progress + 0.0001) : 0.0;

    if (next_price > orderbook_last_price + 0.5) {
        orderbook_last_price = next_price - 0.5;
    }
    else if (next_price < orderbook_last_price) {
        orderbook_last_price = next_price;
    }

    if (contracts > 0.0) {
        // Buy
        execute_order(contracts, orderbook_last_price + 0.5, fee);
    }
    else if (contracts < 0.0) {
        // Sell
        execute_order(contracts, orderbook_last_price, fee);
    }
}

void BitmexSimulator::limit_order(double contracts, double price)
{
    limit_order(contracts, price, true);
}

void BitmexSimulator::limit_order(double contracts, double price, bool use_fee)
{
    const auto next_price = intervals->rows[(long)(intervals_idx + 1)].last_price;
    const auto price_sign = contracts / std::abs(contracts);
    const auto order_price = price - price_sign * 0.5;
    const auto fee = use_fee ? BitSim::BitMex::maker_fee : 0.0;
    //std::cout << "limit_order size(" << contracts << ") price(" << price << ")" << ") order_price(" << order_price << ")" << std::endl;

    if ((contracts > 0.0 && next_price < order_price) ||
        (contracts < 0.0 && next_price > order_price)) {
        execute_order(contracts, price, fee);
        //std::cout << "limit_order size(" << contracts << ")" << std::endl;
    }
}

void BitmexSimulator::execute_order(double order_contracts, double price, double fee)
{
    // Fee
    wallet -= fee * abs(order_contracts / price);

    // Realised profit and loss
    // Wallet only changes when abs(contracts) decrease
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

bool BitmexSimulator::is_liquidated(void)
{
    const auto next_price = intervals->rows[(long)(intervals_idx + 1)].last_price;
    if ((pos_contracts > 0.0 && next_price < liquidation_price()) ||
        (pos_contracts < 0.0 && next_price > liquidation_price())) {
        return true;
    }
    return false;
}
*/
