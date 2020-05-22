#include "pch.h"

#include "Utils.h"
#include "DateTime.h"
#include "BitmexSimulator.h"


BitmexSimulator::BitmexSimulator(sptrIntervals intervals, torch::Tensor features) :
    intervals(intervals),
    features(features),
    intervals_idx_start(0), intervals_idx_end(0),
    intervals_idx(0),
    wallet(0.0), pos_price(0.0), pos_contracts(0.0),
    get_reward_previous_value(0.0)
{

}

sptrRL_State BitmexSimulator::reset(int idx_episode)
{
    logger = std::make_unique<BitmexSimulatorLogger>("bitmex_sim_" + std::to_string(idx_episode) + ".csv", true);

    constexpr auto episode_length = ((std::chrono::seconds) BitSim::Trader::episode_length).count() / ((std::chrono::seconds) BitSim::interval).count();
    intervals_idx = Utils::random(0, (int)intervals->rows.size() - BitSim::observation_length - episode_length - 1); // -2 used for "next step" in training

    // REMOVE
    //intervals_idx = 0;
    const auto intervals_idx_max = std::min((int)intervals->rows.size(), 10 * 60 * 24);
    intervals_idx = Utils::random(0, intervals_idx_max - BitSim::observation_length - episode_length - 1);

    intervals_idx_start = intervals_idx;
    intervals_idx_end = intervals_idx + episode_length;
    
    wallet = 1.0;
    pos_price = 0.0;
    pos_contracts = 0.0;
    start_value = wallet * intervals->rows[intervals_idx_start].last_price;
    get_reward_previous_value = 0.0;
    
    constexpr auto reward = 0.0;
    constexpr auto leverage = 0.0;
    auto state = std::make_shared<RL_State>(reward, features[intervals_idx][0], leverage);
    return state;
}

sptrRL_State BitmexSimulator::step(sptrRL_Action action, bool last_step)
{
    auto timer = Timer{};

    const auto prev_interval = intervals->rows[intervals_idx];

    //action->leverage;
    //action->limit_order;
    //action->market_order;

    auto order_contracts = 0.0;

    if (!action->idle) {

        order_contracts = calculate_order_size(action->leverage);

        //std::cout << "--- " << std::endl;
        //std::cout << "action.buy_size: " << action.buy_size << std::endl;
        //std::cout << "action.sell_size: " << action.sell_size << std::endl;
        //std::cout << "buy_n_contracts: " << buy_contracts << std::endl;
        //std::cout << "sell_n_contracts: " << sell_contracts << std::endl;
        //std::cout << "--- " << std::endl;

        if (action->market_order) {
            //std::cout << "Market order" << std::endl;
            market_order(order_contracts);
        }
        else if (action->limit_order) {
            //std::cout << "Limit order" << std::endl;
            limit_order(order_contracts, prev_interval.last_price);
        }
    }

    const auto reward = get_reward();
    auto [_position_margin, position_leverage, upnl] = calculate_position_leverage(prev_interval.last_price);
    auto state = std::make_shared<RL_State>(reward, features[intervals_idx][0], position_leverage);

    logger->log(prev_interval.last_price,
        wallet,
        upnl,
        pos_contracts,
        position_leverage,
        order_contracts,
        action->leverage,
        action->idle,
        action->limit_order,
        action->market_order,
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
        const auto taker_fee = BitSim::BitMex::taker_fee * abs(pos_contracts / next_price);
        position_pnl = pos_contracts * (1 / pos_price - 1 / next_price) - taker_fee;
    }
    const auto value = (wallet + position_pnl) * next_price / start_value;
    if (get_reward_previous_value == 0.0) {
        get_reward_previous_value = value;
    }
    const auto reward = std::log(value / get_reward_previous_value) * 1000 - 1; // (*1000-1 to get a suitable reward range, between -1000 and -300)
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


    /*
    auto buy_size = 0.0;
    auto sell_size = 0.0;

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
    */
}

void BitmexSimulator::market_order(double contracts)
{
    //std::cout << "market_order size(" << contracts << ")" << std::endl;
    const auto next_price = intervals->rows[(long)(intervals_idx + 1)].last_price;

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
    const auto next_price = intervals->rows[(long)(intervals_idx + 1)].last_price;
    const auto price_sign = contracts / std::abs(contracts);
    const auto order_price = price - price_sign * 0.5;
    //std::cout << "limit_order size(" << contracts << ") price(" << price << ")" << ") order_price(" << order_price << ")" << std::endl;

    if ((contracts > 0.0 && next_price < order_price) ||
        (contracts < 0.0 && next_price > order_price)) {
        execute_order(contracts, price, false);
        //std::cout << "limit_order size(" << contracts << ")" << std::endl;
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

bool BitmexSimulator::is_liquidated(void)
{
    const auto next_price = intervals->rows[(long)(intervals_idx + 1)].last_price;
    if ((pos_contracts > 0.0 && next_price < liquidation_price()) ||
        (pos_contracts < 0.0 && next_price > liquidation_price())) {
        return true;
    }
    return false;
}

BitmexSimulatorLogger::BitmexSimulatorLogger(const std::string &filename, bool enabled) : enabled(enabled)
{
    if (enabled) {
        file.open(std::string{ BitSim::tmp_path } + "\\log\\" + filename);
        file << "last_price,"
            << "wallet,"
            << "upnl,"
            << "position_contracts,"
            << "position_leverage,"
            << "order_contracts,"
            << "order_leverage,"
            << "order_idle,"
            << "order_limit,"
            << "order_market,"
            << "reward" << "\n";
    }
}

void BitmexSimulatorLogger::log(
    double last_price, 
    double wallet, 
    double upnl,
    double position_contracts,
    double position_leverage,
    double order_contracts,
    double order_leverage,
    int order_idle,
    int order_limit,
    int order_market,
    double reward
)
{
    if (enabled) {
        file << last_price << ","
            << wallet << "," 
            << upnl << ","
            << position_contracts << ","
            << position_leverage << ","
            << order_contracts << ","
            << order_leverage << ","
            << order_idle << ","
            << order_limit << ","
            << order_market << ","
            << reward << "\n";
    }
}
