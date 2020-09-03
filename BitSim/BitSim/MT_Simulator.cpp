#include "pch.h"
#include "MT_Simulator.h"
#include "BitLib/BitBotConstants.h"
#include "BitLib/Utils.h"

using namespace std::chrono_literals;


MT_OrderBookBuffer::MT_OrderBookBuffer(void) :
    length(0),
    next_idx(0),
    order_book_bottom(0.0)
{

}

void MT_OrderBookBuffer::step(time_point_ms timestamp, double price)
{
    if (price > order_book_bottom + 0.5) {
        order_book_bottom = price - 0.5;
    }
    else if (price < order_book_bottom) {
        order_book_bottom = price;
    }

    timestamps[next_idx] = timestamp;
    prices[next_idx] = price;

    length = std::min(length + 1, size);
    next_idx = (next_idx + 1) % size;
}

std::tuple<double, double> MT_OrderBookBuffer::get_price(time_point_ms timestamp)
{
    auto count = 0;
    auto idx = (next_idx - 1 + size) % size;
    auto price_bot = std::numeric_limits<double>::max();
    auto price_top = std::numeric_limits<double>::min();
    while (count < length) {
        price_bot = std::min(price_bot, prices[idx]);
        price_top = std::max(price_top, prices[idx]);

        if (timestamps[idx] < timestamp) {
            break;
        }
        idx = (idx - 1 + size) % size;
        ++count;
    }
    if (price_top < order_book_bottom + 0.5) {
        price_top = order_book_bottom + 0.5;
    }
    else if (price_bot > order_book_bottom) {
        price_bot = order_book_bottom;
    }

    return std::make_tuple(price_bot, price_top);
}

MT_OrderBook::MT_OrderBook(time_point_ms timestamp, double price)
{
    buffer.step(timestamp, price);
}

bool MT_OrderBook::update(time_point_ms timestamp, double price, MT_Direction direction)
{
    const auto [price_bot, price_top] = buffer.get_price(timestamp - 1500ms);
    buffer.step(timestamp, price);
    if (direction == MT_Direction::down && price > price_top) {
        return true;
    }
    else if (direction == MT_Direction::up && price < price_bot) {
        return true;
    }
    return false;
}

MT_Simulator::MT_Simulator(const Tick& first_tick) :
    wallet(1.0), pos_price(0.0), pos_contracts(0.0),
    time_since_leverage_change(0.0),
    order_book(first_tick.timestamp, first_tick.price)
{
    // Random leverage at start
    const auto start_leverage = Utils::random_choice({ -BitSim::BitMex::max_leverage, BitSim::BitMex::max_leverage });
    market_order(calculate_order_size(start_leverage), false);
}

void MT_Simulator::step(const Tick& tick)
{
    order_book.update(tick.timestamp, tick.price, tick.buy);

    /*

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
    */
}

void MT_Simulator::market_order(double contracts)
{

}

void MT_Simulator::market_order(double contracts, bool use_fee)
{

}

void MT_Simulator::limit_order(double contracts, double price)
{

}

void MT_Simulator::limit_order(double contracts, double price, bool use_fee)
{

}

void MT_Simulator::execute_order(double contracts, double price, double fee)
{

}

double MT_Simulator::calculate_order_size(double leverage)
{
    return 0.0;
}

std::tuple<double, double, double> MT_Simulator::calculate_position_leverage(double mark_price)
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


/*

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
